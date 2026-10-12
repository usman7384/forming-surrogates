"""Monitoring (task P5): drift tests on force features, and the rolling acceptance metric with a Wilson interval.

Drift: compare the last `window` parts with a reference (the first `reference` parts of the shift) using the
Population Stability Index (quantile bins from the reference) and a two-sample Kolmogorov-Smirnov test. An alert
needs BOTH to fire: PSI alone is noisy at 50 vs 100 samples (its null mean is about (bins-1)(1/n+1/m)), KS alone
fires on tiny shifts that do not matter. This is a design choice, not a standard.

Acceptance: share of measured parts whose prediction is within a tolerance of the measurement. Labels are sparse
(one part in k) and late, so the count is small and a Wilson interval is shown instead of a bare percentage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def psi(reference, current, bins: int = 5, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins of the reference (open-ended outer bins)."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    p = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    q = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_pvalue(reference, current) -> float:
    return float(stats.ks_2samp(reference, current).pvalue)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion: (estimate, low, high). n = 0 gives (nan, 0, 1)."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class DriftMonitor:
    features: tuple = ("f_zero_kN", "peak_force_norm")
    primary: str = "f_zero_kN"          # the feature whose alert counts as THE drift alert
    reference: int = 100
    window: int = 50
    stride: int = 10
    psi_alert: float = 0.25
    ks_alert: float = 1e-3
    bins: int = 5
    # optional effect-size gate per feature: also require |median(window) - median(reference)| > min_shift[feature].
    # Needed for features whose healthy spread is tiny and differs between lots (a 0.05% spread is "significant"
    # to PSI/KS at any lot change but means nothing in practice).
    min_shift: dict = field(default_factory=dict)
    history: dict = field(default_factory=lambda: {})
    alerts: list = field(default_factory=list)       # one dict per evaluation that fired
    evaluations: list = field(default_factory=list)  # every evaluation (for the dashboard)

    def add(self, part_index: int, values: dict) -> list[dict]:
        """Record one part's features; evaluate every `stride` parts once a full window past the reference exists."""
        for k in self.features:
            self.history.setdefault(k, []).append(float(values[k]))
        n = len(self.history[self.features[0]])
        if n < self.reference + self.window or (n - self.reference - self.window) % self.stride:
            return []
        fired = []
        ev = dict(part_index=part_index, n=n)
        for k in self.features:
            h = np.asarray(self.history[k])
            ok_ref, ok_cur = h[:self.reference], h[-self.window:]
            ok_ref, ok_cur = ok_ref[np.isfinite(ok_ref)], ok_cur[np.isfinite(ok_cur)]
            if len(ok_ref) < 20 or len(ok_cur) < 20:
                continue
            ev[f"psi_{k}"], ev[f"ks_p_{k}"] = psi(ok_ref, ok_cur, self.bins), ks_pvalue(ok_ref, ok_cur)
            ev[f"shift_{k}"] = float(np.median(ok_cur) - np.median(ok_ref))
            if (ev[f"psi_{k}"] > self.psi_alert and ev[f"ks_p_{k}"] < self.ks_alert
                    and abs(ev[f"shift_{k}"]) > self.min_shift.get(k, 0.0)):
                a = dict(part_index=part_index, feature=k, psi=ev[f"psi_{k}"], ks_p=ev[f"ks_p_{k}"],
                         primary=(k == self.primary))
                fired.append(a)
        self.evaluations.append(ev)
        self.alerts.extend(fired)
        return fired

    def first_alert(self, feature: str | None = None) -> int | None:
        for a in self.alerts:
            if feature is None or a["feature"] == feature:
                return a["part_index"]
        return None


@dataclass
class AcceptanceTracker:
    """Prediction-vs-measurement bookkeeping. `seen_at` is the stream position (part index) at which the label
    arrived, so a window means "what the operator could know at that time"."""
    tol: float = 1.0
    rows: list = field(default_factory=list)

    def add(self, part_id: int, seen_at: int, predicted: float, measured: float) -> bool:
        acc = bool(np.isfinite(predicted) and abs(predicted - measured) <= self.tol)
        self.rows.append(dict(part_id=part_id, seen_at=seen_at, predicted=predicted, measured=measured,
                              residual=predicted - measured, accepted=acc))
        return acc

    def window(self, now: int, parts: int) -> tuple[int, int]:
        """(accepted, measured) among labels that arrived in the last `parts` parts up to `now`."""
        sel = [r for r in self.rows if now - parts < r["seen_at"] <= now]
        return sum(r["accepted"] for r in sel), len(sel)

    def interval(self, now: int, parts: int):
        return wilson(*self.window(now, parts))

    def span(self, lo: int, hi: int):
        """(accepted, measured) among labels for parts with lo <= part_id < hi."""
        sel = [r for r in self.rows if lo <= r["part_id"] < hi]
        return sum(r["accepted"] for r in sel), len(sel)
