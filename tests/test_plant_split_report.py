"""plant.split_report: per-split acceptance sums back to the demo's headline counts (committed GP results)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "results" / "plant"


@pytest.mark.skipif(not (ROOT / "by_split.json").exists(), reason="results not generated")
def test_by_split_sums_to_headline():
    by = json.loads((ROOT / "by_split.json").read_text())["metrics"]
    for sc in ("drift", "nominal"):
        head = json.loads((ROOT / sc / "metrics.json").read_text())["metrics"]["acceptance"]["all"]
        parts = by[sc]["acceptance_all"].values()
        assert sum(p["accepted"] for p in parts) == head["accepted"]
        assert sum(p["measured"] for p in parts) == head["measured"]
        assert sum(by[sc]["parts_per_split"].values()) == 600
