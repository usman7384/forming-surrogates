"""Failure-to-repro bundle (task P7): everything R&D needs to re-run the failing part in the solver.

A bundle is a directory:
    bundle.json   model name/version, config hash, prediction vs measurement, friction posterior, features, drift state
    signals.parquet  the raw signal window of the part (all three channels, with quality flags)
    case.json     a ringsim case at the INFERRED inputs (ring from the ticket, friction = posterior mean when the
                  prediction was made, reduction = estimate from the signals), in the same layout as a dataset run
    plot.png      force vs die position with the detected contact, and prediction vs measurement
    rerun.sh-style command stored in bundle.json["rerun_command"]; run it with `python -m plant.rerun_bundle <dir>`

Why the inferred inputs and not the truth: the question R&D asks is "was the model wrong, or were the inputs we
fed it wrong?". Re-running the solver at the inputs the edge service believed answers exactly that.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PlantConfig


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True, cwd=Path(__file__).resolve().parents[1]).strip()
    except Exception:
        return "no-git"


def inferred_case(ticket: dict, mf: float, reduction: float) -> dict:
    """ringsim RingCase kwargs at the inferred inputs. K and V are 1 because shape results do not depend on them."""
    return dict(od_id=float(ticket["od_id"]), h_id=float(ticket["h_id"]), mf=float(np.clip(mf, 0.0, 1.0)),
                reduction=float(reduction), m=float(ticket["m"]), K=1.0, V=1.0, n=0.0, eps0=0.01, friction="tresca")


def write_bundle(out_dir: Path, *, rec: dict, ticket: dict, streams: dict, label: dict, cfg: PlantConfig,
                 model_name: str, model_version: str, drift_alerts: list, reason: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [pd.DataFrame(dict(signal=name, t=t, value=v, quality=q)) for name, (t, v, q) in streams.items()]
    pd.concat(rows, ignore_index=True).to_parquet(out_dir / "signals.parquet", index=False)

    case = inferred_case(ticket, rec["mf_mean"], rec["reduction_est"])
    numerics = dict(nz=cfg.bundle_nz)
    (out_dir / "case.json").write_text(json.dumps(dict(case=case, numerics=numerics), indent=2), encoding="utf-8")

    cmd = f"python -m plant.rerun_bundle {out_dir.as_posix()}"
    info = dict(
        reason=reason, part_id=int(rec["part_id"]), lot_id=int(rec["lot_id"]), model=model_name,
        model_version=model_version, model_spec=cfg.model, config_hash=cfg.hash(), config=asdict(cfg), git=_git_sha(),
        prediction=dict(mean_id_change_pct=rec["pred_mean"], std_id_change_pct=rec["pred_std"], verdict=rec["verdict"]),
        measurement=dict(id_change_pct=label["id_change_pct"], sigma_pct=label["sigma_pct"]),
        residual_pct=float(rec["pred_mean"] - label["id_change_pct"]), accept_tol_pct=cfg.accept_tol_pct,
        friction_posterior_at_prediction=dict(mean=rec["mf_mean"], std=rec["mf_std"]),
        features={k: rec[k] for k in ("reduction_est", "f_zero_kN", "peak_kN", "peak_force_norm", "contact_pos_mm",
                                      "contact_anomaly")},
        ticket=ticket, drift_alerts_so_far=drift_alerts[-5:],
        inferred_inputs=case, rerun_command=cmd,
        note="Inputs are what the edge service believed, not ground truth. Re-run to separate model error from input error.")
    (out_dir / "bundle.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    _plot(out_dir / "plot.png", streams, rec, label, cfg)
    return out_dir


def _plot(path: Path, streams: dict, rec: dict, label: dict, cfg: PlantConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from serving.edge.align import align
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    al = align(streams)
    ax[0].plot(al["pos"], al["force"], lw=0.8)
    if np.isfinite(rec["contact_pos_mm"]):
        ax[0].axvline(rec["contact_pos_mm"], color="r", ls="--", lw=0.8, label="detected contact")
    ax[0].set_xlabel("die position (mm)")
    ax[0].set_ylabel("die force (kN)")
    ax[0].set_title(f"part {rec['part_id']}: force vs position")
    ax[0].legend(fontsize=7)
    ax[1].errorbar([0], [rec["pred_mean"]], yerr=[2 * rec["pred_std"]], fmt="o", label="prediction (2 sd)")
    ax[1].errorbar([1], [label["id_change_pct"]], yerr=[2 * label["sigma_pct"]], fmt="s", label="measured (2 sd)")
    ax[1].set_xticks([0, 1], ["predicted", "measured"])
    ax[1].set_ylabel("inner-diameter change (%)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
