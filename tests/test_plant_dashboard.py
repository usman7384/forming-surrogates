"""P6: dashboard data helpers and a headless run of the Streamlit script on a small offline shift."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from dashboard.data import acceptance_at, acceptance_series, drift_status, label_status, load_results
from plant.config import PlantConfig
from plant.demo import compute_metrics
from plant.press_sim import PressSimulator
from serving.edge.service import EdgeService


@pytest.fixture(scope="module")
def shift_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("shift")
    cfg = PlantConfig(scenario="drift", n_parts=200, lot_len=40, drift_slope_kN_per_part=0.2)
    sim, svc = PressSimulator(cfg), EdgeService(cfg)
    svc.bundle_dir = out / "bundles"
    for i in range(cfg.n_parts):
        for lab in sim.labels_due(i):
            svc.on_label(lab)
        p = sim.part(i)
        svc.process_part(p.ticket, p.streams, time.time_ns())
    svc.save(out)
    (out / "metrics.json").write_text(json.dumps({"metrics": compute_metrics(cfg, svc, sim, 1.0)}, default=float))
    return out


def test_results_load_and_acceptance_helpers(shift_dir):
    R = load_results(shift_dir)
    assert len(R.parts) == 200 and not R.labels.empty
    k, n, p, lo, hi = acceptance_at(R.labels, now=199, window=100)
    assert 0 < n <= 21 and lo <= p <= hi
    s = acceptance_series(R.labels, 200, 100)
    assert {"rate", "low", "high"} <= set(s.columns)
    ls = label_status(R.parts, R.labels, now=60, label_every=5)
    assert ls["sampled"] == 12 and ls["measured"] + ls["waiting"] == ls["sampled"]
    ds = drift_status(R.events, now=199)
    assert ds["first"] is None or ds["first"] >= 100


def test_streamlit_script_runs_headless(shift_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(sys, "argv", ["app.py", "--results", str(shift_dir)])
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Latest verdict" in labels and any(l.startswith("Acceptance") for l in labels)
