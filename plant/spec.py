"""Loader for docs/signal-interface.yaml, the single source of truth for tag names and process constants.

Why: the simulator, server, edge service and tests all read the same file, so a renamed tag cannot silently
break one of them (tests/test_plant_spec.py enforces it).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "signal-interface.yaml"


@dataclass(frozen=True)
class Signal:
    name: str
    browse_name: str          # e.g. "Press1.Stroke.Force" (the string part of the node id)
    node_id: str
    unit: str
    rate_hz: float | None     # None for event signals
    is_chunk: bool
    raw: dict


@dataclass(frozen=True)
class Spec:
    interface: dict
    process: dict
    chunk: dict
    signals: dict[str, Signal]

    @property
    def namespace_uri(self) -> str:
        return self.interface["namespace_uri"]

    @property
    def endpoint(self) -> str:
        return self.interface["endpoint_default"]


@lru_cache(maxsize=4)
def load_spec(path: str | None = None) -> Spec:
    doc = yaml.safe_load(Path(path or SPEC_PATH).read_text(encoding="utf-8"))
    sigs = {}
    for s in doc["signals"]:
        rate = s["sampling_rate_hz"]
        sigs[s["name"]] = Signal(
            name=s["name"], browse_name=s["node_id"].split(";s=", 1)[1], node_id=s["node_id"], unit=s["unit"],
            rate_hz=float(rate) if isinstance(rate, (int, float)) else None,
            is_chunk=str(s["data_type"]).startswith("Double[]"), raw=s)
    return Spec(doc["interface"], doc["process_assumptions"], doc["chunk_format"], sigs)
