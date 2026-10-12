"""Task N3 sanity check: can the model memorise 2 graphs? Writes results/gnn/n3_overfit_<data>.json.

    python -m surrogates.gnn.overfit --data smoke --out results/gnn
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .results import write_result
from .train import fit, load_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", default="results/gnn")
    ap.add_argument("--config", default="surrogates/gnn/configs/overfit2.yaml")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    with tempfile.TemporaryDirectory() as td:
        fit(cfg, a.data, Path(td), resume=False)
        rows = [json.loads(l) for l in open(Path(td) / "log.jsonl")]
    first, last = rows[0], rows[-1]
    metrics = {"epochs": len(rows), "n_graphs": cfg.overfit_samples,
               "first": {k: first[k] for k in ("train_total", "train_disp", "train_strain", "train_force")},
               "final": {k: last[k] for k in ("train_total", "train_disp", "train_strain", "train_force")},
               "note_units": "losses are MSE on normalised targets (1.0 = predicting the training mean)"}
    cmd = "python -m surrogates.gnn.overfit " + " ".join(sys.argv[1:])
    p = write_result(f"{a.out}/n3_overfit_{a.data}.json", "N3", "Overfit two graphs", cmd,
                     {"dataset": a.data, "n_runs": cfg.overfit_samples}, metrics,
                     notes=f"config {a.config}; two (run, step) samples from different runs; lambda_j={cfg.lambda_j}")
    print(p, metrics["final"])


if __name__ == "__main__":
    main()
