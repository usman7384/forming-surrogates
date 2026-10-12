"""OPC-UA server (asyncua) that publishes the simulated press as the tags listed in docs/signal-interface.yaml.

Run standalone:
    python -m plant.opcua_server --scenario drift --parts 600 --port 4840

Security is OFF (NoSecurity, anonymous): this is a localhost demo, not a hardened deployment.
The simulated press waits for the client handshake tag `Press1.Control.EdgeReady` before the first stroke.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
from asyncua import Server, ua

from .chunks import encode_chunks
from .config import PlantConfig
from .press_sim import Part, PressSimulator
from .spec import Spec, load_spec

STATE = {"idle": 0, "approach": 1, "forming": 2, "hold": 3, "retract": 4}
_SIGNAL_OF_STREAM = {"stroke_position": "stroke_position", "stroke_force": "stroke_force",
                     "die_temperature": "die_temperature"}


def _variant_for(sig) -> ua.Variant:
    dt = str(sig.raw["data_type"])
    if dt.startswith("Double[]"):
        return ua.Variant([0.0], ua.VariantType.Double)
    if dt == "Int32":
        return ua.Variant(0, ua.VariantType.Int32)
    if dt == "Boolean":
        return ua.Variant(False, ua.VariantType.Boolean)
    return ua.Variant("", ua.VariantType.String)


class PressServer:
    def __init__(self, endpoint: str | None = None, spec: Spec | None = None):
        self.spec = spec or load_spec()
        self.endpoint = endpoint or self.spec.endpoint
        self.server = Server()
        self.nodes: dict = {}
        self.n_labels = 0                       # labels published so far, stamped into every stroke_end
        self.idx = None

    async def start(self) -> None:
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        self.idx = await self.server.register_namespace(self.spec.namespace_uri)
        objs: dict[str, object] = {}
        for sig in self.spec.signals.values():
            parts = sig.browse_name.split(".")
            parent, path = self.server.nodes.objects, ""
            for p in parts[:-1]:
                path = f"{path}.{p}" if path else p
                if path not in objs:
                    objs[path] = await parent.add_object(ua.NodeId(path, self.idx), ua.QualifiedName(p, self.idx))
                parent = objs[path]
            var = await parent.add_variable(ua.NodeId(sig.browse_name, self.idx),
                                            ua.QualifiedName(parts[-1], self.idx), _variant_for(sig))
            if sig.name == "edge_ready":
                await var.set_writable()
            self.nodes[sig.name] = var
        await self.server.start()

    async def stop(self) -> None:
        await self.server.stop()

    # ------------------------------------------------------------------ writes
    async def _w(self, name: str, value, vtype: ua.VariantType) -> None:
        await self.nodes[name].write_value(ua.Variant(value, vtype))

    async def wait_for_edge(self, timeout_s: float = 60.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if await self.nodes["edge_ready"].read_value():
                return True
            await asyncio.sleep(0.05)
        return False

    async def publish_part(self, part: Part) -> None:
        chunk_s = float(self.spec.chunk["chunk_duration_s"])
        await self._w("part_ticket", json.dumps(part.ticket), ua.VariantType.String)
        await self._w("stroke_state", STATE["approach"], ua.VariantType.Int32)
        chunks = {name: encode_chunks(part.part_id, *part.streams[name], part.t_start, chunk_s)
                  for name in _SIGNAL_OF_STREAM}
        await self._w("stroke_state", STATE["forming"], ua.VariantType.Int32)
        for k in range(max(len(c) for c in chunks.values())):       # interleave streams, as a PLC block scan would
            for name, cl in chunks.items():
                if k < len(cl):
                    await self._w(name, cl[k].tolist(), ua.VariantType.Double)
        await self._w("stroke_state", STATE["retract"], ua.VariantType.Int32)
        end = dict(part_id=part.part_id, n_chunks={n: len(c) for n, c in chunks.items()}, t_end_s=part.t_end,
                   emit_wall_ns=time.time_ns(), labels_published=self.n_labels)
        await self._w("stroke_state", STATE["idle"], ua.VariantType.Int32)
        await self._w("stroke_end", json.dumps(end), ua.VariantType.String)

    async def publish_label(self, label: dict) -> None:
        self.n_labels += 1
        msg = {k: label[k] for k in ("part_id", "id_change_pct", "sigma_pct", "reduction", "due_index")}   # due_index: server stroke counter
        await self._w("lab_measurement", json.dumps(msg), ua.VariantType.String)

    async def run_shift(self, sim: PressSimulator, pace_s: float = 0.0, wait_edge: bool = True) -> None:
        if wait_edge and not await self.wait_for_edge():
            raise TimeoutError("edge service never signalled EdgeReady")
        for i in range(sim.cfg.n_parts):
            for lab in sim.labels_due(i):
                await self.publish_label(lab)
            await self.publish_part(sim.part(i))
            if pace_s > 0:
                await asyncio.sleep(pace_s)


async def serve(cfg: PlantConfig, endpoint: str | None, linger_s: float = 2.0) -> None:
    srv = PressServer(endpoint)
    await srv.start()
    print(f"[server] listening on {srv.endpoint}", flush=True)
    try:
        await srv.run_shift(PressSimulator(cfg), cfg.pace_s)
        print("[server] shift finished", flush=True)
        await asyncio.sleep(linger_s)          # let the client drain its notifications
    finally:
        await srv.stop()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML with PlantConfig fields")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--parts", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--pace-s", type=float, default=None)
    ap.add_argument("--endpoint", default=None)
    a = ap.parse_args(argv)
    cfg = PlantConfig.from_yaml(a.config) if a.config else PlantConfig()
    for k, v in (("scenario", a.scenario), ("n_parts", a.parts), ("seed", a.seed), ("dataset", a.dataset),
                 ("pace_s", a.pace_s)):
        if v is not None:
            setattr(cfg, k, v)
    asyncio.run(serve(cfg, a.endpoint))


if __name__ == "__main__":
    main()
