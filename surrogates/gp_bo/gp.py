"""Exact Gaussian process (Matern 5/2, ARD): trained with GPyTorch, served with plain numpy.

Why two halves: GPyTorch is convenient for maximising the marginal likelihood, but the plant's edge
service should not need torch to answer a request. After training, a GP posterior is fully described by
(training inputs, training targets, kernel hyper-parameters), so `NumpyGP` rebuilds the Cholesky factor
from those few numbers and predicts with numpy only. A test checks it matches GPyTorch.

Inputs are scaled to [0, 1] with the *fixed* box below (the campaign's training box plus the reduction
range), so ARD length scales of different inputs are comparable: length scale 0.3 means "the target
changes appreciably over 30% of the input's training range".
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

# fixed input box, FEATURES order: od_id, h_id, mf, m, reduction  (campaign training box; reduction 10-30%)
LO = np.array([1.6, 0.4, 0.0, 0.05, 0.10])
HI = np.array([2.4, 1.0, 1.0, 0.30, 0.30])
FEATURES = ["od_id", "h_id", "mf", "m", "reduction"]
SQRT5 = math.sqrt(5.0)
DEFAULT_THREADS = 4          # pinned: the fit is deterministic for a given thread count, platform and package versions


def to_unit(X: np.ndarray, lo: np.ndarray = LO, hi: np.ndarray = HI) -> np.ndarray:
    return (np.asarray(X, float) - lo) / (hi - lo)


def matern52(A: np.ndarray, B: np.ndarray, ls: np.ndarray, outputscale: float) -> np.ndarray:
    a, b = A / ls, B / ls
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    r = np.sqrt(np.maximum(d2, 0.0))
    return outputscale * (1.0 + SQRT5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-SQRT5 * r)


@dataclass
class NumpyGP:
    """Posterior of an exact GP, evaluated with numpy. Targets are standardised internally."""
    Xu: np.ndarray            # (n, d) training inputs in unit box
    y: np.ndarray             # (n,) raw training targets
    ls: np.ndarray            # (d,) length scales (unit-box units)
    outputscale: float        # signal variance on standardised targets
    noise: float              # noise variance on standardised targets
    mean: float               # constant mean on standardised targets
    lo: np.ndarray = None     # type: ignore[assignment]
    hi: np.ndarray = None     # type: ignore[assignment]
    names: tuple = tuple(FEATURES)

    def __post_init__(self):
        self.lo = LO if self.lo is None else np.asarray(self.lo)
        self.hi = HI if self.hi is None else np.asarray(self.hi)
        self.ymu, self.ysd = float(self.y.mean()), float(self.y.std() + 1e-12)
        K = matern52(self.Xu, self.Xu, self.ls, self.outputscale) + self.noise * np.eye(len(self.Xu))
        self._L = np.linalg.cholesky(K)
        z = (self.y - self.ymu) / self.ysd - self.mean
        self._alpha = solve_triangular(self._L.T, solve_triangular(self._L, z, lower=True), lower=False)

    def predict(self, X: np.ndarray, include_noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and std (original target units) at raw inputs X (n, d)."""
        Xu = to_unit(np.atleast_2d(X), self.lo, self.hi)
        Ks = matern52(Xu, self.Xu, self.ls, self.outputscale)
        mu = Ks @ self._alpha + self.mean
        v = solve_triangular(self._L, Ks.T, lower=True)   # O(n^2) per query, not a full solve
        var = self.outputscale - (v ** 2).sum(0) + (self.noise if include_noise else 0.0)
        return mu * self.ysd + self.ymu, np.sqrt(np.maximum(var, 1e-12)) * self.ysd

    # ---- persistence: only the numbers needed to rebuild the posterior
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, Xu=self.Xu, y=self.y, ls=self.ls, lo=self.lo, hi=self.hi,
                            scalars=np.array([self.outputscale, self.noise, self.mean]),
                            names=np.array(self.names))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyGP":
        z = np.load(path)
        s = z["scalars"]
        return cls(Xu=z["Xu"], y=z["y"], ls=z["ls"], outputscale=float(s[0]), noise=float(s[1]),
                   mean=float(s[2]), lo=z["lo"], hi=z["hi"], names=tuple(str(n) for n in z["names"]))


def fit_gp(X: np.ndarray, y: np.ndarray, *, lo: np.ndarray = LO, hi: np.ndarray = HI,
           names=tuple(FEATURES), seed: int = 0, n_restarts: int = 3, iters: int = 300,
           max_ls: float = 1e3, verbose: bool = False, threads: int = DEFAULT_THREADS) -> tuple[NumpyGP, dict]:
    """Fit an exact Matern-5/2 ARD GP by maximising the marginal likelihood (Adam, best of restarts).

    Returns (NumpyGP, info) where info has the learned hyper-parameters and the final log-likelihood.
    The noise is bounded below at 1e-6 (standardised units): the simulator is deterministic, so the only
    "noise" is discretisation roughness; a floor keeps the Cholesky stable.
    """
    import gpytorch
    import torch
    from gpytorch.constraints import GreaterThan, Interval

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    Xu = torch.tensor(to_unit(X, lo, hi), dtype=torch.float64)
    ymu, ysd = float(np.mean(y)), float(np.std(y) + 1e-12)
    yt = torch.tensor((np.asarray(y, float) - ymu) / ysd, dtype=torch.float64)
    d = Xu.shape[1]

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, xt, yt_, lik):
            super().__init__(xt, yt_, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=d, lengthscale_constraint=Interval(0.02, max_ls)))

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    best = None
    rng = np.random.default_rng(seed)
    for r in range(n_restarts):
        lik = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
        model = _GP(Xu, yt, lik).double()
        ls0 = 0.5 if r == 0 else float(rng.uniform(0.2, 2.0))
        model.covar_module.base_kernel.lengthscale = torch.full((1, d), ls0, dtype=torch.float64)
        model.covar_module.outputscale = 1.0
        lik.noise = 1e-3
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.08)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for i in range(iters):
            opt.zero_grad()
            loss = -mll(model(Xu), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = float(-mll(model(Xu), yt))
        if verbose:
            print(f"  restart {r}: -mll/n = {final:.4f}")
        if best is None or final < best[0]:
            best = (final, model, lik)
    final, model, lik = best
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel().copy()
    gp = NumpyGP(Xu=Xu.numpy(), y=np.asarray(y, float), ls=ls,
                 outputscale=float(model.covar_module.outputscale.item()),
                 noise=float(lik.noise.item()), mean=float(model.mean_module.constant.item()),
                 lo=lo, hi=hi, names=tuple(names))
    info = dict(neg_mll_per_point=final, lengthscales=dict(zip(names, map(float, ls))),
                outputscale=gp.outputscale, noise=gp.noise, n_train=int(len(y)), seed=seed,
                env=environment(threads),
                _torch=(model, lik, Xu, yt, ymu, ysd))
    return gp, info


def environment(threads: int) -> dict:
    """Versions and thread count that make a fit reproducible (recorded in artifacts/gp/meta.json)."""
    import gpytorch
    import scipy
    import torch
    return dict(torch=torch.__version__, gpytorch=gpytorch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, torch_threads=int(threads))


def public_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def dump_json(path: str | Path, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
