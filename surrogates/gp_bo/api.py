"""G6: serve the GP behind the interfaces of `serving/model_api.py`.

    surrogate, estimator = load("artifacts/gp")

`Surrogate.predict` evaluates the cached GP posterior with numpy only (no torch at serving time).
`FrictionEstimator` is an exact grid posterior over m_f that uses the GP as forward model; the GP's error is
treated as one offset shared by all labels of a lot (see serving/model_api.GridFrictionEstimator), so a
poorly-known region of input space widens the posterior instead of producing false confidence.

Which sd the plant sees: `artifacts/gp/meta.json` field `served_sd` is "conformal" (GP sd times the
split-conformal factor `sd_scale.id`) or "raw" (the GP's own sd). The choice is made by a fixed rule in
`surrogates.gp_bo.run` and reported in results/gp/gp_accuracy.json (`served_sd`).

`in_data_support(X)` is NOT part of the frozen Surrogate contract: it flags queries in the censored corner
(high friction and thin rings at high reduction, where early fold-over left only survivor runs); see
`surrogates/gp_bo/support.py` for the rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from serving.model_api import FEATURES, GridFrictionEstimator
from surrogates.gp_bo.gp import NumpyGP
from surrogates.gp_bo.support import in_support, load_support

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "gp"
_SUPPORT_CACHE: dict[str, dict] = {}


class GPSurrogate:
    """Inner-diameter change (%) from the five scalar-model inputs."""

    def __init__(self, gp: NumpyGP, name: str = "gp-matern52-ard", version: str = "0", sd_scale: float = 1.0,
                 support: dict | None = None):
        self.gp, self.name, self.version = gp, name, version
        self.sd_scale, self.support = float(sd_scale), support

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} columns in order {FEATURES}, got {X.shape[1]}")
        mu, sd = self.gp.predict(X)
        return mu, sd * self.sd_scale

    def in_data_support(self, X: np.ndarray) -> np.ndarray:
        if self.support is None:
            return np.ones(len(np.atleast_2d(X)), bool)
        return in_support(X, self.support)


def in_data_support(X: np.ndarray, artifact_dir: str | None = None) -> np.ndarray:
    """False where the query lies in the censored corner of the training data (see support.py)."""
    d = str(Path(artifact_dir) if artifact_dir else DEFAULT_DIR)
    if d not in _SUPPORT_CACHE:
        _SUPPORT_CACHE[d] = load_support(Path(d) / "support.json")
    return in_support(X, _SUPPORT_CACHE[d])


def load(artifact_dir: str | None = None):
    """Return (Surrogate, FrictionEstimator) from `<artifact_dir>/id_gp.npz`, `meta.json`, `support.json`."""
    d = Path(artifact_dir) if artifact_dir else DEFAULT_DIR
    gp = NumpyGP.load(d / "id_gp.npz")
    meta = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
    scale = meta.get("sd_scale", {}).get("id", 1.0) if meta.get("served_sd") == "conformal" else 1.0
    support = load_support(d / "support.json") if (d / "support.json").exists() else None
    s = GPSurrogate(gp, name=meta.get("name", "gp-matern52-ard"), version=str(meta.get("version", "0")),
                    sd_scale=scale, support=support)
    return s, GridFrictionEstimator(s, n=201)
