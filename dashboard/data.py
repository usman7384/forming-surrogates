"""Data access for the operator dashboard, kept free of Streamlit so it can be unit-tested."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from serving.edge.monitoring import wilson


@dataclass
class Results:
    parts: pd.DataFrame
    labels: pd.DataFrame
    events: list
    drift: pd.DataFrame
    metrics: dict
    bundles: list
    root: Path


def load_results(root: str | Path) -> Results:
    r = Path(root)
    events = [json.loads(l) for l in (r / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bundles = []
    for b in sorted((r / "bundles").glob("part_*/bundle.json")):
        info = json.loads(b.read_text(encoding="utf-8"))
        bundles.append(dict(path=b.parent, part_id=info["part_id"], reason=info["reason"],
                            residual=info["residual_pct"], rerun=info["rerun_command"]))
    meas = r / "measurements.parquet"
    return Results(parts=pd.read_parquet(r / "parts.parquet"),
                   labels=pd.read_parquet(meas) if meas.exists() else pd.DataFrame(),
                   events=events, drift=pd.read_parquet(r / "drift.parquet"),
                   metrics=json.loads((r / "metrics.json").read_text(encoding="utf-8")), bundles=bundles, root=r)


def acceptance_at(labels: pd.DataFrame, now: int, window: int) -> tuple[int, int, float, float, float]:
    """(accepted, measured, rate, wilson_low, wilson_high) among labels that arrived in (now-window, now]."""
    if labels.empty:
        return 0, 0, float("nan"), 0.0, 1.0
    sel = labels[(labels.part_index > now - window) & (labels.part_index <= now)]
    k, n = int(sel.accepted.sum()), len(sel)
    p, lo, hi = wilson(k, n)
    return k, n, p, lo, hi


def acceptance_series(labels: pd.DataFrame, n_parts: int, window: int, step: int = 5) -> pd.DataFrame:
    rows = []
    for now in range(window, n_parts, step):
        k, n, p, lo, hi = acceptance_at(labels, now, window)
        rows.append(dict(part=now, rate=p, low=lo, high=hi, n_labels=n))
    return pd.DataFrame(rows)


def label_status(parts: pd.DataFrame, labels: pd.DataFrame, now: int, label_every: int) -> dict:
    """What the operator can and cannot know at part `now`: sampled parts still waiting for their measurement."""
    done = set(labels[labels.part_index <= now].part_id) if not labels.empty else set()
    sampled = [p for p in parts.part_id[parts.part_id <= now] if p % label_every == label_every - 1]
    waiting = [p for p in sampled if p not in done]
    return dict(sampled=len(sampled), measured=len(done), waiting=len(waiting), waiting_parts=waiting[-10:],
                unsampled_share=1 - len(sampled) / max(1, now + 1))


def drift_status(events: list, now: int, primary: str = "f_zero_kN") -> dict:
    al = [e for e in events if e["type"] == "drift_alert" and e["part_index"] <= now]
    prim = [e for e in al if e["feature"] == primary]
    return dict(active=bool(prim and now - prim[-1]["part_index"] <= 30), first=prim[0]["part_index"] if prim else None,
                last=prim[-1]["part_index"] if prim else None, n_primary=len(prim), n_other=len(al) - len(prim))
