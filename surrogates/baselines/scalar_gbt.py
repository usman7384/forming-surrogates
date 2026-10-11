"""Baseline 2 for the scalar targets: gradient-boosted regression trees.

Why: a strong generic tabular learner. Trees cannot extrapolate (they predict a constant outside the
training range), so the out-of-distribution geometry split is where it should struggle.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class GBTBaseline:
    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                               subsample=0.9, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBTBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
