"""P1/P3: the OPC-UA server publishes exactly the tags in docs/signal-interface.yaml and a client can read them."""
import asyncio
import json
import socket

import numpy as np
from asyncua import Client

from plant.chunks import decode_chunk
from plant.config import PlantConfig
from plant.opcua_server import PressServer
from plant.press_sim import PressSimulator
from plant.spec import load_spec


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_spec_node_ids_are_consistent():
    spec = load_spec()
    assert len(spec.signals) >= 8
    for s in spec.signals.values():
        assert s.node_id == f"ns=2;s={s.browse_name}", s.name
    assert spec.signals["die_temperature"].raw["description"].count("NUISANCE") == 1


def test_client_reads_all_tags_and_a_published_part():
    spec = load_spec()
    sim = PressSimulator(PlantConfig(n_parts=3, scenario="nominal"))
    ep = f"opc.tcp://127.0.0.1:{_free_port()}/t/"

    async def go():
        srv = PressServer(ep)
        await srv.start()
        try:
            assert srv.idx == 2, "spec assumes our namespace index is 2"
            async with Client(ep) as cl:
                nodes = {n: cl.get_node(s.node_id) for n, s in spec.signals.items()}
                for n, node in nodes.items():          # every spec tag exists and is readable
                    await node.read_value()
                await srv.publish_part(sim.part(0))
                ticket = json.loads(await nodes["part_ticket"].read_value())
                end = json.loads(await nodes["stroke_end"].read_value())
                pid, seq, t, v, q = decode_chunk(await nodes["stroke_force"].read_value())
                return ticket, end, (pid, seq, t, v, q)
        finally:
            await srv.stop()

    ticket, end, (pid, seq, t, v, q) = asyncio.run(go())
    assert ticket["part_id"] == end["part_id"] == pid == 0
    assert set(end["n_chunks"]) == {"stroke_position", "stroke_force", "die_temperature"}
    assert seq == end["n_chunks"]["stroke_force"] - 1          # last force chunk is the one we read
    assert len(t) == len(v) == len(q) > 0 and np.all(np.diff(t) >= 0)
