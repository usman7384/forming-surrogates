"""Re-run a repro bundle's ringsim case.   python -m plant.rerun_bundle <bundle_dir> [--nz N]

Prints the solver's inner-diameter change at the bundle's inferred inputs next to the model's prediction and the
measurement, and writes rerun.json into the bundle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ringsim.solver import Numerics, RingCase, RingSolver


def rerun(bundle_dir: str | Path, nz: int | None = None) -> dict:
    d = Path(bundle_dir)
    spec = json.loads((d / "case.json").read_text(encoding="utf-8"))
    info = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    num = dict(spec.get("numerics", {}))
    if nz is not None:
        num["nz"] = nz
    res = RingSolver(RingCase(**spec["case"]), Numerics(**num)).run()
    out = dict(solver_id_change_pct=float(res.qoi["id_change_pct"]),
               final_reduction=float(res.qoi["final_reduction"]),
               stopped_early=bool(res.info.get("stopped_early")),
               model_prediction_pct=info["prediction"]["mean_id_change_pct"],
               measured_pct=info["measurement"]["id_change_pct"], numerics=num, case=spec["case"])
    out["solver_minus_measured_pct"] = out["solver_id_change_pct"] - out["measured_pct"]
    out["solver_minus_prediction_pct"] = out["solver_id_change_pct"] - out["model_prediction_pct"]
    (d / "rerun.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--nz", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(rerun(a.bundle, a.nz), indent=2))


if __name__ == "__main__":
    main()
