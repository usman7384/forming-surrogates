"""Starter solver for the ring compression test.

Physics (isothermal, rigid-viscoplastic, incompressible):
  * Norton-Hoff / power-law flow stress   sigma_bar = K * eps_dot_eq**m   (optional hardening (eps0 + eps)**n)
  * Levy-Mises flow rule                   s = (2/3) * sigma_bar / eps_dot_eq * eps_dot
  * Regularised Tresca friction on the die tau = -mf * sigma_bar/sqrt(3) * v_r / sqrt(v_r**2 + delta**2)
  * Velocity-pressure mixed formulation, Taylor-Hood P2/P1 triangles, axisymmetric (r, z)
  * Half model: mid-plane z = 0 is a symmetry plane (v_z = 0), the top die moves down at speed V
  * Updated Lagrangian time stepping, Newton iterations with the consistent tangent of the power law; the friction tangent holds the flow stress fixed
    (its velocity dependence is left out), which slows convergence slightly but does not change the converged solution
  * Die force from the power balance: F * V = 2*pi * (int 2*eta*eps:eps r dA + int c*v_r**2 r ds)

Everything is written in plain numpy/scipy + scikit-fem so it installs with pip on any OS.
Lengths are dimensionless: the initial inner radius is 1.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.sparse import bmat, coo_matrix
from skfem import (BilinearForm, Basis, ElementTriP1, ElementTriP2, ElementVector,
                   FacetBasis, MeshTri, condense, solve)

SOLVER_VERSION = "0.1.0"


# --------------------------------------------------------------------------- inputs
@dataclass
class RingCase:
    """One ring compression run. Geometry and friction are dimensionless.

    od_id      outer diameter / inner diameter (standard 6:3:2 ring -> 2.0)
    h_id       height / inner diameter         (standard 6:3:2 ring -> 2/3)
    mf         Tresca friction factor, 0 (frictionless) .. 1 (sticking); ignored if friction='noslip'
    reduction  final height reduction, e.g. 0.4 for 40 %
    m          strain-rate sensitivity of the material (0 < m <= 1)
    K          material consistency (only scales the force; shape results do not depend on it)
    V          die speed (only scales the force through V**m; shape results do not depend on it)
    n, eps0    optional strain hardening sigma_bar = K (eps0 + eps)^n eps_dot^m  (n = 0: none)
    friction   'tresca' or 'noslip' (sticking bound used for verification)
    """
    od_id: float = 2.0
    h_id: float = 2.0 / 3.0
    mf: float = 0.3
    reduction: float = 0.4
    m: float = 0.15
    K: float = 1.0
    V: float = 1.0
    n: float = 0.0
    eps0: float = 0.01
    friction: str = "tresca"


@dataclass
class Numerics:
    nz: int = 8                 # elements through the half height
    dh_frac: float = 0.01       # die travel per step, as a fraction of the initial half height
    save_every: int = 2         # save fields every k steps
    eps_reg: float = 1e-3       # strain-rate regularisation, relative to V / h0
    delta_rel: float = 1e-2     # friction regularisation velocity, relative to V
    newton_tol: float = 1e-8    # relative velocity update
    newton_max: int = 60
    contact_tol: float = 1e-6   # relative to the initial half height
    min_area_frac: float = 0.05 # stop the run if an element shrinks below this fraction of the mean area
    mesh: str = "structured"    # 'structured' (tensor triangles) or 'unstructured' (jittered Delaunay)
    mesh_seed: int = 0          # RNG seed of the unstructured point cloud (ignored if structured)
    mesh_jitter: float = 0.25   # interior point jitter as a fraction of the spacing (unstructured only)


@dataclass
class RunResult:
    case: dict
    numerics: dict
    qoi: dict
    curves: dict
    fields: dict = field(repr=False)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- mesh
def make_mesh(Ri: float, Ro: float, h0: float, nz: int, kind: str = "structured",
              seed: int = 0, jitter: float = 0.25) -> MeshTri:
    """Mesh of the half ring cross-section [Ri, Ro] x [0, h0] with about nz elements through h0.

    'structured' is the original tensor mesh (default, frozen). 'unstructured' builds a jittered
    lattice and triangulates it with Delaunay: boundary points stay exactly on the boundary
    (only slid along it, corners fixed) so the contact and symmetry conditions are unchanged, while
    interior points are displaced by up to jitter*spacing in both directions. The domain is convex,
    so every boundary edge is a Delaunay edge. Why: the structured mesh aligns element edges with
    the flow; a surrogate trained on it could be learning the mesh. Used for the test_unstructured split.
    """
    nr = max(2, int(math.ceil((Ro - Ri) / (h0 / nz))))
    if kind == "structured":
        return MeshTri.init_tensor(np.linspace(Ri, Ro, nr + 1), np.linspace(0.0, h0, nz + 1))
    if kind != "unstructured":
        raise ValueError(f"unknown mesh kind {kind!r}")
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    dr, dz = (Ro - Ri) / nr, h0 / nz
    R, Z = np.meshgrid(np.linspace(Ri, Ro, nr + 1), np.linspace(0.0, h0, nz + 1), indexing="ij")
    R, Z = R.copy(), Z.copy()
    inner = np.zeros(R.shape, bool)
    inner[1:-1, 1:-1] = True
    R[inner] += rng.uniform(-jitter, jitter, inner.sum()) * dr
    Z[inner] += rng.uniform(-jitter, jitter, inner.sum()) * dz
    # slide boundary points along their edge (corners stay fixed)
    for sl, axis_d, arr, dd in ((np.s_[1:-1, 0], 0, R, dr), (np.s_[1:-1, -1], 0, R, dr),
                                (np.s_[0, 1:-1], 1, Z, dz), (np.s_[-1, 1:-1], 1, Z, dz)):
        arr[sl] += rng.uniform(-jitter, jitter, arr[sl].shape) * dd
    pts = np.column_stack([R.ravel(), Z.ravel()])
    tri = Delaunay(pts).simplices
    return MeshTri(np.ascontiguousarray(pts.T), np.ascontiguousarray(tri.T))


# --------------------------------------------------------------------------- forms
def _eps(u, r):
    """Axisymmetric strain-rate components (rr, zz, tt, rz) of a vector field."""
    g = u.grad
    return g[0][0], g[1][1], u[0] / r, 0.5 * (g[0][1] + g[1][0])


_G = np.array([1.0, 1.0, 1.0, 2.0])     # metric of the (rr, zz, tt, rz) double contraction


class _Geometry:
    """Basis-function strain rates and quadrature weights on one mesh, for fast numpy assembly.

    The volume terms (viscous secant, Newton tangent, divergence) are assembled with einsum
    instead of scikit-fem's per-basis-pair Python callbacks.
    """

    def __init__(self, bu, bp):
        r = bu.global_coordinates()[0]                      # (nel, nq)
        nb = bu.element_dofs.shape[0]
        S = np.empty((nb, 4) + r.shape)
        for i in range(nb):
            phi = bu.basis[i][0]
            g, v = phi.grad, phi
            S[i, 0], S[i, 1], S[i, 2], S[i, 3] = g[0][0], g[1][1], v[0] / r, 0.5 * (g[0][1] + g[1][0])
        self.S = S
        self.SG = S * _G[None, :, None, None]
        self.W = bu.dx * r                                         # r-weighted quadrature weights
        self.edofs = bu.element_dofs                               # (nb, nel)
        self.N = bu.N
        self.rows = np.broadcast_to(self.edofs[:, None, :], (nb, nb, r.shape[0])).ravel()
        self.cols = np.broadcast_to(self.edofs[None, :, :], (nb, nb, r.shape[0])).ravel()
        # divergence block B (rows: pressure dofs, cols: velocity dofs): -int q div(u) r dA
        psi = np.array([np.asarray(bp.basis[k][0]) for k in range(bp.element_dofs.shape[0])])
        divS = S[:, 0] + S[:, 1] + S[:, 2]
        Bloc = -np.einsum("keq,ieq,eq->kie", psi, divS, self.W, optimize=True)
        prow = np.broadcast_to(bp.element_dofs[:, None, :], Bloc.shape).ravel()
        pcol = np.broadcast_to(self.edofs[None, :, :], Bloc.shape).ravel()
        self.B = coo_matrix((Bloc.ravel(), (prow, pcol)), shape=(bp.N, bu.N)).tocsr()

    def strain(self, u):
        return np.einsum("ie,iceq->ceq", u[self.edofs], self.S, optimize=True)

    def matrix(self, local):
        return coo_matrix((local.ravel(), (self.rows, self.cols)), shape=(self.N, self.N)).tocsr()

    def viscous(self, coef):
        """int coef * e(u):e(v) r dA   (coef = 2 eta)."""
        return self.matrix(np.einsum("iceq,jceq,eq->ije", self.SG, self.S, coef * self.W,
                                     optimize=True))

    def tangent(self, coef, e):
        """int coef * (e0:e(u)) (e0:e(v)) r dA."""
        P = np.einsum("iceq,ceq->ieq", self.SG, e, optimize=True)
        return self.matrix(np.einsum("ieq,jeq,eq->ije", P, P, coef * self.W, optimize=True))


@BilinearForm
def _friction(u, v, w):
    return w.c * u[0] * v[0] * w.x[0]


def _eq_rate(ufield, r):
    err, ezz, ett, erz = _eps(ufield, r)
    return np.sqrt(2.0 / 3.0 * (err ** 2 + ezz ** 2 + ett ** 2 + 2.0 * erz ** 2))


# --------------------------------------------------------------------------- solver
class RingSolver:
    def __init__(self, case: RingCase, num: Numerics | None = None):
        self.case = case
        self.num = num or Numerics()
        c = case
        self.Ri0 = 1.0
        self.Ro0 = c.od_id * self.Ri0
        self.h0 = c.h_id * 2.0 * self.Ri0 / 2.0      # half height  (H = h_id * ID, ID = 2 Ri)
        nz = self.num.nz
        self.mesh = make_mesh(self.Ri0, self.Ro0, self.h0, nz, self.num.mesh,
                              self.num.mesh_seed, self.num.mesh_jitter)
        self.t = self.mesh.t.copy()
        self.eu = ElementVector(ElementTriP2())
        self.ep = ElementTriP1()
        self.strain = np.zeros(self.mesh.nelements)      # cumulated equivalent strain (per element)
        self.rate0 = c.V / self.h0                        # nominal strain rate

    # ---- material
    def _flow_stress(self, rate_reg, strain_qp):
        c = self.case
        hard = (c.eps0 + strain_qp) ** c.n if c.n else 1.0
        return c.K * hard * rate_reg ** c.m

    # ---- one velocity solve on the current configuration
    def _solve_step(self, mesh, z_die, u_guess):
        c, num = self.case, self.num
        bu = Basis(mesh, self.eu, intorder=4)
        bp = Basis(mesh, self.ep, intorder=4)
        top = mesh.facets_satisfying(lambda x: x[1] >= z_die - num.contact_tol * self.h0,
                                     boundaries_only=True)
        bottom = mesh.facets_satisfying(lambda x: x[1] <= num.contact_tol * self.h0,
                                        boundaries_only=True)
        fb = FacetBasis(mesh, self.eu, facets=top, intorder=4) if len(top) else None

        D_top = bu.get_dofs(top).all("u^2")
        D_bot = bu.get_dofs(bottom).all("u^2")
        D = [D_top, D_bot]
        x = np.zeros(bu.N + bp.N)
        x[D_top] = -c.V
        if c.friction == "noslip":
            D_r = bu.get_dofs(top).all("u^1")
            D.append(D_r)
        D = np.unique(np.concatenate(D))

        r_qp = bu.global_coordinates()[0]
        strain_qp = np.repeat(self.strain[:, None], r_qp.shape[1], axis=1)
        rate_floor = num.eps_reg * self.rate0
        delta = num.delta_rel * c.V
        geo = _Geometry(bu, bp)
        B = geo.B

        fric_on = fb is not None and c.friction == "tresca" and c.mf > 0
        if fric_on:
            r_f = fb.global_coordinates()[0]
            f_strain = np.repeat(self.strain[fb.tind][:, None], r_f.shape[1], axis=1)

        def operators(u):
            """Secant operator A(u) (A(u) u = internal forces) and Newton tangent Kt(u)."""
            e = geo.strain(u)
            q = np.sqrt(2.0 / 3.0 * (e[0] ** 2 + e[1] ** 2 + e[2] ** 2 + 2.0 * e[3] ** 2)
                        + rate_floor ** 2)
            eta = self._flow_stress(q, strain_qp) / (3.0 * q)
            A = geo.viscous(2.0 * eta)
            Kt = A + geo.tangent((4.0 / 3.0) * (c.m - 1.0) * eta / q ** 2, e)
            if fric_on:
                fuf = fb.interpolate(u)
                fe = _eps(fuf, r_f)
                fq = np.sqrt(2.0 / 3.0 * (fe[0] ** 2 + fe[1] ** 2 + fe[2] ** 2 + 2.0 * fe[3] ** 2)
                             + rate_floor ** 2)
                tau_y = c.mf * self._flow_stress(fq, f_strain) / math.sqrt(3.0)
                s = np.sqrt(fuf[0] ** 2 + delta ** 2)
                A = A + _friction.assemble(fb, c=tau_y / s)
                Kt = Kt + _friction.assemble(fb, c=tau_y * delta ** 2 / s ** 3)
            return A, Kt, q

        # Newton iterations (consistent tangent of the power law and of the friction law,
        # flow stress frozen inside the friction tangent), backtracking on the residual norm
        u = (u_guess if u_guess is not None else self._initial_guess(bu)).copy()
        u[D] = x[D]
        p = np.zeros(bp.N)
        free = np.ones(bu.N + bp.N, dtype=bool)
        free[D] = False
        it, err, alpha, r_old, du_last = 0, np.inf, 1.0, np.inf, None
        for it in range(1, num.newton_max + 1):
            A, Kt, q = operators(u)
            R = np.concatenate([A @ u + B.T @ p, B @ u])
            r_norm = np.linalg.norm(R[free])
            if r_norm > r_old and du_last is not None and alpha > 1.0 / 64:
                alpha *= 0.5                       # backtrack the previous step
                u -= alpha * du_last[: bu.N]
                p -= alpha * du_last[bu.N:]
                continue
            r_old, alpha = r_norm, 1.0
            J = bmat([[Kt, B.T], [B, None]], format="csr")
            d = solve(*condense(J, -R, D=D))
            u += d[: bu.N]
            p += d[bu.N:]
            du_last = d
            err = np.linalg.norm(d[: bu.N]) / max(np.linalg.norm(u), 1e-30)
            if err < num.newton_tol:
                break

        # power balance -> die force (2*pi for the full revolution; half model = one die)
        A, _, q = operators(u)
        power = u @ (A @ u)
        force = 2.0 * math.pi * power / c.V
        rate_el = q.mean(axis=1)
        return dict(u=u, p=p, bu=bu, force=force, iters=it, err=err, rate_el=rate_el,
                    converged=bool(err < num.newton_tol))

    def _initial_guess(self, bu):
        """Homogeneous compression field: v_r = V r / (2h), v_z = -V z / h."""
        h = self.mesh.p[1].max()
        xy = bu.doflocs
        u = np.zeros(bu.N)
        u[bu.nodal_dofs[0]] = self.case.V * xy[0, bu.nodal_dofs[0]] / (2 * h)
        u[bu.nodal_dofs[1]] = -self.case.V * xy[1, bu.nodal_dofs[1]] / h
        u[bu.facet_dofs[0]] = self.case.V * xy[0, bu.facet_dofs[0]] / (2 * h)
        u[bu.facet_dofs[1]] = -self.case.V * xy[1, bu.facet_dofs[1]] / h
        return u

    def _areas(self, p):
        tri = p[:, self.t]
        return 0.5 * ((tri[0, 1] - tri[0, 0]) * (tri[1, 2] - tri[1, 0])
                      - (tri[0, 2] - tri[0, 0]) * (tri[1, 1] - tri[1, 0]))

    def _volume(self, p):
        rc = p[0, self.t].mean(axis=0)
        return 2.0 * math.pi * np.sum(np.abs(self._areas(p)) * rc)   # Pappus, exact for P1 triangles

    def _quality_ok(self, p):
        """False when an element collapses or inverts (fold-over onto the die needs remeshing)."""
        a = self._areas(p)
        return bool(np.all(np.sign(a) == self._area_sign) and
                    np.min(np.abs(a)) > self.num.min_area_frac * self._area_ref)

    # ---- full stroke
    def run(self, verbose: bool = False) -> RunResult:
        c, num = self.case, self.num
        t0 = time.time()
        p = self.mesh.p.copy()
        vol0 = self._volume(p)
        self._inner = np.isclose(p[0], self.Ri0)
        a0 = self._areas(p)
        self._area_sign, self._area_ref = np.sign(a0), np.mean(np.abs(a0))
        stopped_early = False
        dh = num.dh_frac * self.h0
        n_steps = int(round(c.reduction * self.h0 / dh))
        dt = dh / c.V
        z_die = self.h0

        stroke, force, iters, id_curve, idmin_curve = [0.0], [], [], [], []
        newton_errs, n_unconv = [], 0
        saved = dict(step=[], p=[], strain=[], rate=[], vel=[], press=[])
        u = None
        warn = []
        for k in range(n_steps + 1):
            mesh = MeshTri(p, self.t)
            res = self._solve_step(mesh, z_die, u)
            u = res["u"]
            newton_errs.append(res["err"])
            if not res["converged"]:
                n_unconv += 1
                warn.append(f"step {k}: Newton not converged to tolerance "
                            f"(err={res['err']:.1e} after {res['iters']} iterations)")
            force.append(res["force"])
            iters.append(res["iters"])
            id_curve.append(100.0 * (p[0, np.isclose(p[1], 0.0)].min() - self.Ri0) / self.Ri0)
            idmin_curve.append(100.0 * (p[0, self._inner].min() - self.Ri0) / self.Ri0)
            bu = res["bu"]

            def save_state():
                saved["step"].append(k)
                saved["p"].append(p.copy())
                saved["strain"].append(self.strain.copy())
                saved["rate"].append(res["rate_el"].copy())
                saved["vel"].append(np.vstack([u[bu.nodal_dofs[0]], u[bu.nodal_dofs[1]]]))
                saved["press"].append(res["p"].copy())

            if k % num.save_every == 0 or k == n_steps:
                save_state()
            if k == n_steps:
                break
            # updated Lagrangian move, midpoint rule (2nd order): predict half a step, re-solve,
            # then move the full step with the mid-step velocity and accumulate the mid-step strain rate
            v0 = np.vstack([u[bu.nodal_dofs[0]], u[bu.nodal_dofs[1]]])
            p_half = p + 0.5 * dt * v0
            z_half = z_die - 0.5 * dh
            p_half[1] = np.clip(p_half[1], 0.0, z_half)
            if not self._quality_ok(p_half):
                stopped_early = True
                warn.append(f"stopped at step {k}: element collapse (fold-over), remeshing needed")
                if saved["step"][-1] != k:
                    save_state()
                break
            res_h = self._solve_step(MeshTri(p_half, self.t), z_half, u)
            uh, buh = res_h["u"], res_h["bu"]
            newton_errs.append(res_h["err"])
            if not res_h["converged"]:
                n_unconv += 1
                warn.append(f"step {k}+1/2: Newton not converged to tolerance "
                            f"(err={res_h['err']:.1e} after {res_h['iters']} iterations)")
            p_new = p + dt * np.vstack([uh[buh.nodal_dofs[0]], uh[buh.nodal_dofs[1]]])
            p_new[1] = np.clip(p_new[1], 0.0, z_die - dh)        # no penetration into the die
            if not self._quality_ok(p_new):
                stopped_early = True
                warn.append(f"stopped at step {k}: element collapse (fold-over), remeshing needed")
                if saved["step"][-1] != k:
                    save_state()
                break
            p = p_new
            self.strain = self.strain + dt * res_h["rate_el"]
            z_die -= dh
            u = uh
            stroke.append((k + 1) * dh)
            if verbose and (k + 1) % 10 == 0:
                print(f"  step {k+1}/{n_steps}  F={force[-1]:.4g}  newton={iters[-1]}")

        stroke = np.array(stroke)
        force = np.array(force)
        mid = np.isclose(p[1], 0.0)
        top = np.isclose(p[1], z_die)
        ri_mid = p[0, mid].min()
        ro_mid = p[0, mid].max()
        ro_top = p[0, top].max() if top.any() else np.nan
        A0 = math.pi * (self.Ro0 ** 2 - self.Ri0 ** 2)
        f_scale = c.K * self.rate0 ** c.m * A0
        vol = self._volume(p)
        qoi = dict(
            id_change_pct=100.0 * (ri_mid - self.Ri0) / self.Ri0,          # at mid-height
            id_min_change_pct=100.0 * (p[0, self._inner].min() - self.Ri0) / self.Ri0,
            id_die_change_pct=100.0 * (p[0, top].min() - self.Ri0) / self.Ri0 if top.any() else np.nan,
            od_change_pct=100.0 * (ro_mid - self.Ro0) / self.Ro0,
            barreling=(ro_mid - ro_top) / self.Ro0,
            peak_force=float(force.max()),
            peak_force_norm=float(force.max() / f_scale),
            max_strain=float(self.strain.max()),
            final_reduction=float(stroke[-1] / self.h0),
            volume_drift_pct=100.0 * (vol - vol0) / vol0,
        )
        info = dict(solver_version=SOLVER_VERSION, runtime_s=round(time.time() - t0, 2),
                    n_steps=n_steps, newton_iters_max=int(max(iters)),
                    newton_unconverged_solves=int(n_unconv),
                    newton_err_max=float(max(newton_errs)),
                    newton_iters_mean=float(np.mean(iters)), warnings=warn,
                    stopped_early=stopped_early,
                    n_nodes=int(p.shape[1]), n_elements=int(self.t.shape[1]))
        curves = dict(stroke=stroke, reduction=stroke / self.h0, force=force,
                      force_norm=force / f_scale, id_change_pct=np.array(id_curve),
                      id_min_change_pct=np.array(idmin_curve))
        fields = {k: np.array(v) for k, v in saved.items()}
        fields["t"] = self.t
        numerics = asdict(num)
        if num.mesh == "structured":      # keep the stored numerics identical to schema v1 by default
            for k in ("mesh", "mesh_seed", "mesh_jitter"):
                numerics.pop(k)
        return RunResult(case=asdict(c), numerics=numerics, qoi=qoi, curves=curves,
                         fields=fields, info=info)


def homogeneous_force(case: RingCase, reduction: float) -> float:
    """Exact frictionless die force: F = K (V/h)^m * pi (Ro^2 - Ri^2) with current h and radii."""
    Ri0, Ro0 = 1.0, case.od_id
    h0 = case.h_id
    h = h0 * (1.0 - reduction)
    s = math.sqrt(h0 / h)
    return case.K * (case.V / h) ** case.m * math.pi * ((Ro0 * s) ** 2 - (Ri0 * s) ** 2)


def save_run(res: RunResult, path_stem: str) -> None:
    """Write <stem>.npz (fields + curves) and <stem>.json (inputs, outputs, provenance)."""
    arrays = {f"field_{k}": v for k, v in res.fields.items()}
    arrays.update({f"curve_{k}": v for k, v in res.curves.items()})
    np.savez_compressed(path_stem + ".npz", **arrays)
    meta = dict(case=res.case, numerics=res.numerics, qoi=res.qoi, info=res.info)
    with open(path_stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
