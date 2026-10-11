"""Baseline 1 for the scalar targets: ridge-regularised cubic polynomial in the (unit-scaled) inputs.

Why: it is the thing an engineer would try first (a response surface). If the GP cannot beat it, the GP
is not earning its keep.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class PolyBaseline:
    def __init__(self, degree: int = 3, alpha: float = 1e-3):
        self.model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(alpha=alpha))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolyBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
