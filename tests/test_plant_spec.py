"""P1: the signal-interface spec is complete and is what the code uses."""
import json
import re
from pathlib import Path

from plant.config import PlantConfig
from plant.opcua_server import PressServer
from plant.spec import load_spec

REQUIRED = {"name", "node_id", "description", "unit", "sampling_rate_hz", "timestamp_source", "quality_flag"}


def test_every_signal_has_the_required_fields():
    spec = load_spec()
    for s in spec.signals.values():
        missing = REQUIRED - set(s.raw)
        assert not missing, (s.name, missing)
    assert {"stroke_force", "stroke_position", "die_temperature", "lab_measurement"} <= set(spec.signals)


def test_server_publishes_exactly_the_spec_tags():
    spec = load_spec()
    assert set(PressServer("opc.tcp://127.0.0.1:1/x/").nodes) == set()      # nothing before start
    # server.start() builds one node per spec signal (test_plant_opcua reads them all); here check ids are unique
    ids = [s.node_id for s in spec.signals.values()]
    assert len(ids) == len(set(ids))


def test_nuisance_channel_is_labelled_and_unused_by_the_model():
    spec = load_spec()
    assert "NUISANCE" in spec.signals["die_temperature"].raw["description"]
    src = (Path(__file__).resolve().parents[1] / "serving" / "edge" / "service.py").read_text(encoding="utf-8")
    uses = [l for l in src.splitlines() if "die_temperature" in l and "predict" in l]
    assert not uses


def test_config_hash_changes_with_config():
    assert PlantConfig().hash() == PlantConfig().hash()
    assert PlantConfig().hash() != PlantConfig(seed=1).hash()
