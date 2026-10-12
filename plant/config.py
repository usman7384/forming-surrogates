"""One config object for the whole plant stack, hashable so every bundle records which config produced it.

Switching from the stub model to the real GP is a change of `model` / `artifact_dir` here (or on the CLI), nothing else.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class PlantConfig:
    # --- run
    seed: int = 0
    scenario: str = "drift"            # nominal | drift (= drift_offset) | drift_offset | drift_gain | drift_friction
    n_parts: int = 600
    # --- model (config-only switch to the real GP)
    model: str = "serving.model_api:load_stub"
    artifact_dir: str | None = None
    # --- replay data
    dataset: str = "smoke"
    splits: tuple = ("test_random", "test_ood_geometry", "test_fine_mesh")
    lot_len: int = 50                  # parts per lubricant lot (a lot replays one solver run's friction and geometry)
    # --- process scatter (simulated, not from the solver)
    reduction_sd: float = 0.008        # stroke-to-stroke scatter of the achieved reduction
    id_scatter_pct: float = 0.15       # unmodelled part-to-part scatter of the true ID change, percentage points
    k_lot_sd: float = 0.05             # lot-to-lot relative scatter of the flow-stress consistency K
    # --- sensors
    force_noise_kN: float = 0.5
    force_clock_offset_ms: float = 3.0
    # --- drift scenario
    drift_start_frac: float = 0.5
    drift_slope_kN_per_part: float = 0.05       # drift_offset: load-cell zero offset ramp
    drift_gain_per_part: float = 0.002          # drift_gain: load-cell gain error ramp (multiplicative, 1 + g*k)
    # drift_friction: lots made from the drift start on use runs with higher and higher true friction m_f
    friction_drift_q0: float = 0.4              # first drifted lot draws from the runs above this m_f quantile
    friction_drift_q1: float = 0.9              # last drifted lot draws from runs above this quantile
    # --- labels
    label_every: int = 5               # one part in k is measured
    label_delay_parts: tuple = (3, 8)  # measurement arrives this many parts after the stroke (inclusive range)
    label_sigma_pct: float = 0.3       # gauge uncertainty, percentage points
    # --- ticket spec
    tol_id_pct: float = 2.0            # part is in spec if |ID change - target| <= tol
    # --- edge service
    contact_threshold_kN: float = 5.0
    contact_debounce: int = 3
    friction_nodes: float = 1.7320508075688772   # Gauss-Hermite nodes (0, +-sqrt 3), weights (2/3, 1/6, 1/6)
    interval_k: float = 2.0            # verdict uses prediction +- interval_k * std
    recovery: str = "none"             # "none" (baseline) | "rezero": after a zero-offset alert, subtract the estimated offset
    rezero_window: int = 5             # parts averaged for the offset estimate (includes the current part)
    # --- acceptance metric / monitoring
    accept_tol_pct: float = 1.0        # a measured part is "accepted" if |prediction - measurement| <= this
    window_parts: int = 100            # acceptance window and drift window length
    drift_window_parts: int = 50
    drift_stride: int = 10
    reference_parts: int = 100         # drift reference = first N parts of the shift
    psi_alert: float = 0.25
    # --- repro bundles
    max_bundles: int = 20
    bundle_nz: int = 8                 # mesh resolution written into the bundle's ringsim case (8 = campaign default)
    # --- pacing / latency
    pace_s: float = 0.0                # wall-clock pause between parts on the server (0 = as fast as the edge allows)
    assumed_cycle_time_s: float = 6.0

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlantConfig":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k in ("splits", "label_delay_parts"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


DRIFT_TYPES = {"nominal": None, "drift": "offset", "drift_offset": "offset", "drift_gain": "gain",
               "drift_friction": "friction"}


def drift_type(scenario: str) -> str | None:
    """'drift' is kept as the original name of the zero-offset scenario."""
    if scenario not in DRIFT_TYPES:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(DRIFT_TYPES)}")
    return DRIFT_TYPES[scenario]


def load_model(spec: str, artifact_dir: str | None = None):
    """'package.module:function' -> (Surrogate, FrictionEstimator). The only place the model is named."""
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)(artifact_dir)
