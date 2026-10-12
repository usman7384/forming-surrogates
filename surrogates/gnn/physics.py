"""Incompressibility measure J per triangle, in torch so it can be a training loss (tasks N2, N4)."""
from __future__ import annotations

import torch


def j_torch(pos: torch.Tensor, p0: torch.Tensor, tri: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """J = (A/A0)(rbar/rbar0) per triangle. pos, p0: (N, 2) = (r, z) current / initial; tri: (3, T).
    Signed area is used (not abs) so an inverted triangle gives J < 0 and is penalised, not hidden."""
    def ar(p):
        a, b, c = p[tri[0]], p[tri[1]], p[tri[2]]
        area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        return area, (a[:, 0] + b[:, 0] + c[:, 0]) / 3.0
    a, rb = ar(pos)
    a0, rb0 = ar(p0)
    sign = torch.sign(a0)                 # orientation of the initial triangle
    return (a * sign / a0.abs().clamp_min(eps)) * (rb / rb0.clamp_min(eps))
