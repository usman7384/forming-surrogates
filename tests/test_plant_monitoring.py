"""P5: drift tests and the acceptance metric behave on synthetic data."""
import numpy as np
import pytest

from serving.edge.monitoring import AcceptanceTracker, DriftMonitor, ks_pvalue, psi, wilson


def test_wilson_known_values():
    p, lo, hi = wilson(8, 10)
    assert p == 0.8 and lo == pytest.approx(0.490, abs=2e-3) and hi == pytest.approx(0.943, abs=2e-3)
    assert wilson(0, 0)[1:] == (0.0, 1.0)
    p, lo, hi = wilson(0, 20)                      # behaves at the boundary: lower bound 0, upper bound > 0
    assert lo == 0.0 and 0.1 < hi < 0.25
    assert wilson(50, 50)[2] == 1.0


def test_psi_and_ks_detect_a_shift_and_ignore_none():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, rng.normal(0, 1, 2000)) < 0.05
    assert psi(ref, rng.normal(1.0, 1, 2000)) > 0.25
    assert ks_pvalue(ref, rng.normal(0, 1, 500)) > 0.001
    assert ks_pvalue(ref, rng.normal(0.5, 1, 500)) < 1e-6


def _stream(n, drift_at=None, slope=0.02, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, n)
    if drift_at is not None:
        x[drift_at:] += slope * np.arange(1, n - drift_at + 1)
    return x


def test_monitor_alerts_after_drift_not_before():
    mon = DriftMonitor(features=("a",), primary="a", reference=100, window=50, stride=10)
    x = _stream(400, drift_at=250)
    for i, v in enumerate(x):
        mon.add(i, {"a": v})
    first = mon.first_alert("a")
    assert first is not None and 250 <= first <= 300      # fires within one window of the drift start
    assert all(a["part_index"] >= 250 for a in mon.alerts)


def test_monitor_false_alarm_rate_on_null_data_is_low():
    alarms = 0
    for seed in range(30):
        mon = DriftMonitor(features=("a",), primary="a")
        for i, v in enumerate(_stream(500, seed=seed)):
            mon.add(i, {"a": v})
        alarms += bool(mon.alerts)
    assert alarms <= 1                                    # at most 1 of 30 null shifts raises any alert


def test_acceptance_windows_and_interval():
    acc = AcceptanceTracker(tol=1.0)
    for k in range(20):
        acc.add(part_id=k, seen_at=k + 5, predicted=0.0, measured=0.5 if k < 10 else 3.0)
    assert acc.span(0, 10) == (10, 10) and acc.span(10, 20) == (0, 10)
    assert acc.window(now=14, parts=10) == (10, 10)       # labels that arrived at parts 5..14 are ids 0..9
    assert acc.window(now=19, parts=10) == (5, 10)        # arrived at parts 10..19 are ids 5..14
    p, lo, hi = acc.interval(now=24, parts=10)
    assert 0 <= lo <= p <= hi <= 1
