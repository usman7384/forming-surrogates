"""Acceptance by replay split (task P8 follow-up):  python -m plant.split_report --root results/plant

Each lot of the demo replays one solver run, and each run belongs to a split (test_random, test_ood_geometry,
test_fine_mesh). This joins the demo's measurements to that mapping, so the headline acceptance can be read per
split (OOD and fine-mesh parts are harder than in-distribution ones). It re-derives the lot-to-run mapping from the
saved config.yaml (same seed, same pool), so it needs the dataset the demo used but does not re-run the demo.
Writes <root>/by_split.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from serving.edge.monitoring import wilson
from .bundle import _git_sha
from .config import PlantConfig
from .press_sim import PressSimulator


def _rate(df: pd.DataFrame) -> dict:
    k, n = int(df.accepted.sum()), len(df)
    p, lo, hi = wilson(k, n)
    return dict(accepted=k, measured=n, rate=p, wilson95=[lo, hi])


def report(scenario_dir: Path) -> dict:
    cfg = PlantConfig.from_yaml(scenario_dir / "config.yaml")
    sim = PressSimulator(cfg)
    parts = pd.read_parquet(scenario_dir / "parts.parquet")
    lab = pd.read_parquet(scenario_dir / "measurements.parquet")
    lot_split = {int(l): sim.lot(int(l))["run"].split for l in parts.lot_id.unique()}
    parts["split"] = parts.lot_id.map(lot_split)
    lab = lab.merge(parts[["part_id", "split", "lot_id"]], on="part_id")
    lab["before"] = lab.part_id < (sim.drift_start if sim.drift_start is not None else 10**9)
    out = dict(lots_per_split={s: sum(v == s for v in lot_split.values()) for s in cfg.splits},
               parts_per_split={s: int((parts.split == s).sum()) for s in cfg.splits},
               verdicts_per_split={s: parts[parts.split == s].verdict.value_counts().to_dict() for s in cfg.splits},
               median_pred_std_pct_points={s: float(parts[parts.split == s].pred_std.median()) for s in cfg.splits},
               median_pred_std_pct_points_per_lot={int(l): float(g.pred_std.median()) for l, g in parts.groupby("lot_id")},
               lot_split={int(l): v for l, v in lot_split.items()},
               acceptance_all={s: _rate(lab[lab.split == s]) for s in cfg.splits if (lab.split == s).any()})
    if sim.drift_start is not None:
        out["acceptance_before_drift"] = {s: _rate(lab[(lab.split == s) & lab.before])
                                          for s in cfg.splits if ((lab.split == s) & lab.before).any()}
        out["acceptance_after_drift"] = {s: _rate(lab[(lab.split == s) & ~lab.before])
                                         for s in cfg.splits if ((lab.split == s) & ~lab.before).any()}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/plant")
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = {"task": "P8", "title": "Plant demo acceptance by replay split", "git": _git_sha(),
           "command": f"python -m plant.split_report --root {a.root}",
           "metrics": {s: report(root / s) for s in ("drift", "nominal") if (root / s / "config.yaml").exists()},
           "notes": "Counts per split are small (label every 5th part, 12 lots per scenario); read the Wilson intervals."}
    (root / "by_split.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res["metrics"], indent=1, default=float))


if __name__ == "__main__":
    main()
