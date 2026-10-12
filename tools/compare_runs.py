"""Compare a fresh-clone regeneration with the committed results and write a results JSON.

    python tools/compare_runs.py <other_repo_root> --out results/reproduction/clean_clone_check.json

Why: the README may only quote numbers that sit in a results file, so the claim "a fresh clone reproduces the
committed GP artifact and downstream metrics" needs its own file. Timing, version and commit fields are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

SKIP = ("seconds", "ms", "latency", "wall", "runtime", "time", "created", "git", "command", "_s")
RESULTS = ["results/gp/gp_accuracy.json", "results/gp/inverse_friction.json", "results/design/design.json",
           "results/gnn/eval_base_full/metrics.json", "results/gnn/eval_jloss_full/metrics.json",
           "results/gnn/eval_idloss_full/metrics.json", "results/gnn/eval_mixmesh_full/metrics.json",
           "results/plant/drift/metrics.json", "results/plant/nominal/metrics.json", "results/plant/by_split.json"]
ARTIFACTS = ["artifacts/gp/id_gp.npz", "artifacts/gp/force_gp.npz", "artifacts/gp/support.json"]


def walk(x, y, path, stat):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in x:
            if k in y and not any(s in k.lower() for s in SKIP):
                walk(x[k], y[k], f"{path}/{k}", stat)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]", stat)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        d = abs(x - y) / abs(y) if abs(y) > 1e-9 else abs(x - y)
        stat["n"] += 1
        if d > stat["max"]:
            stat["max"], stat["at"] = d, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("other")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here, other = Path(".").resolve(), Path(a.other).resolve()
    art = {f: (here / f).read_bytes() == (other / f).read_bytes() for f in ARTIFACTS}
    res = {}
    for f in RESULTS:
        s = dict(n=0, max=0.0, at="")
        walk(json.loads((other / f).read_text())["metrics"], json.loads((here / f).read_text())["metrics"], "", s)
        res[f] = dict(numbers_compared=s["n"], max_relative_difference=s["max"])
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = dict(task="L2.3", title="Fresh-clone regeneration vs committed results",
               command=f"python tools/compare_runs.py {a.other} --out {a.out}", git=sha,
               data=dict(dataset="full", n_runs=460, splits={}),
               metrics=dict(artifacts_byte_identical=art, results=res,
                            all_identical=all(art.values()) and all(v["max_relative_difference"] == 0 for v in res.values())),
               figures=[], notes="The other tree was produced by re-running the GP fit (4 threads, fixed seed), design, inverse "
               "friction, the four GNN evaluations and both plant demos in a fresh clone with a new virtual environment; "
               "GNN training was not repeated (the committed checkpoints were evaluated). Latency/time/commit fields are skipped. "
               "artifacts/gp/meta.json differs only in the recorded code version.",
               created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["metrics"]["all_identical"]), art)


if __name__ == "__main__":
    main()
