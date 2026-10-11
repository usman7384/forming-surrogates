# ringsim: starter solver for the ring compression test

A small, verified finite-element solver for the **ring compression test** (the standard
forging friction test), plus a campaign runner that turns it into an ML dataset.
Pure Python (numpy, scipy, scikit-fem), installs with `pip` on Windows, macOS and Linux.

## What it solves

Half of a flat ring (mid-plane symmetry) squeezed between flat rigid dies, axisymmetric (r, z):

| Ingredient | Choice |
| --- | --- |
| Material | Isothermal rigid-viscoplastic, Norton–Hoff power law `σ̄ = K ε̇^m` (optional hardening `(ε0+ε)^n`) |
| Incompressibility | Velocity–pressure mixed formulation, Taylor–Hood P2/P1 triangles |
| Friction | Regularised Tresca `τ = −m_f (σ̄/√3) v_r / √(v_r² + δ²)`, `m_f` from 0 (frictionless) to 1 (sticking); exact no-slip option for the sticking bound |
| Nonlinear solve | Newton with the consistent tangent of the power law and of the friction law, backtracking line search |
| Time stepping | Updated Lagrangian, midpoint (2nd-order) rule |
| Die force | Power balance `F·V = ∫ σ:ε̇ dV + ∫ τ·Δv dS` (no noisy stress integration) |
| Contact | Nodes reaching the die are projected onto it and join the contact set |
| Limits | No remeshing: when an element collapses (fold-over at high friction), the run stops cleanly and is flagged |

Lengths are dimensionless (initial inner radius = 1). Because the model is isothermal and the friction
law scales with the flow stress, **shape results do not depend on K or on the die speed V**; force scales
as `K·V^m`. The solver proves this (check V2), so the dataset uses dimensionless inputs only.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                                      # 3 fast regression tests (~10 s)
python -m ringsim.verify --out verify_out               # full verification + calibration chart (~3 min on 2 cores)
python -m ringsim.campaign --out data --n-train 300 --n-ood 40 --n-fine 20   # dataset (resumable)
python -m ringsim.viz data/runs/run_0000                 # plot one run
```

```python
from ringsim.solver import RingCase, RingSolver
res = RingSolver(RingCase(od_id=2.0, h_id=2/3, mf=0.3, m=0.15, reduction=0.35)).run()
print(res.qoi)          # id_change_pct, peak_force_norm, barreling, max_strain, volume_drift_pct, ...
res.curves["force_norm"], res.curves["id_change_pct"]   # per step
res.fields["p"], res.fields["strain"]                   # node positions / element strain at saved steps
```

## Verification results (6:3:2 ring, rate sensitivity 0.15, nz = 8)

| Check | Result |
| --- | --- |
| V1 frictionless homogeneous-field test (called a patch test in the report text): the exact solution lies in the FE space, so this shows only that the discretisation reproduces an exact homogeneous solution; it says nothing about friction or contact (40% reduction) | force error 0.002%, inner-diameter error 0.005%, strain error 0.001%, volume drift 0.002% |
| V2 scaling in K and V implemented and checked (x10 each) | inner-diameter change identical to 1e-13; force ratios 10.000000 and 10^0.15 exactly |
| V3 friction trend at 30% reduction | +19.5% (m=0) → +10.5% (0.1) → +2.1% (0.2) → −4.4% (0.3) → −14.8% (0.5) → −22.2% (1.0), monotonic |
| V4 mesh convergence (m=0.3, 30%) | −4.42% / −4.44% / −4.44% for nz = 6 / 8 / 12 (high friction not yet tested) |
| V5 sticking bound (29%) | Tresca m=1: −21.19%, no-slip: −21.62%; the gap is mostly the friction regularisation (0.07 points with delta_rel = 1e-3) |
| V6 solver health (15 runs) | every Newton solve reached the 1e-8 tolerance (logged per solve; mean 2–14, max 25 iterations); worst volume drift 0.74%; m ≥ 0.7 runs stop at 29–36% (fold-over) |

The friction trend agrees **in trend**, by eye, with the textbook shape of ring calibration charts (no quantitative comparison with published data was made); those charts assume
rigid-plastic material, so exact numbers are not expected to match.

![calibration chart](reference/calibration_chart.png)

## Dataset layout (from `ringsim.campaign`)

```
data/
├── plan.json          # every planned case (inputs, split, mesh) — fixed once created, so runs resume
├── index.csv          # one row per run: inputs, QoIs, split, status, stopped_early, runtime
├── dataset_card.md    # ranges, counts, failures, known limits
├── failures.jsonl     # tracebacks of failed runs
└── runs/run_XXXX.npz  # field_p (steps,2,nodes), field_strain (steps,elements), field_vel, field_press,
    runs/run_XXXX.json #   field_t (triangles), curve_* (per step); json = inputs, QoIs, solver version, git hash
```

Splits: `train`, `test_random` (every 10th Latin hypercube point), `test_ood_geometry` (outer-diameter or height
ratio outside the training box), `test_fine_mesh` (random test points on a 1.5x finer mesh; tests
discretisation dependence, not generalisation).

## Known limitations (honest list for the README of your project)

- **No remeshing.** High-friction or thin rings stop when the bulging side folds onto the die (21–36%
  reduction). The data up to that point is kept and flagged. Scalar QoIs in `index.csv` are at each
  run's FINAL reduction, so build ML targets from the per-step curves at fixed reductions.
- **Contact never releases**, which is fine for compression.
- **Volume drift** is always slightly negative (up to 0.74% at high friction). It is a spatial error
  (shrinks with mesh refinement, not with the time step) from moving the P1 geometry with vertex
  velocities and from clipping nodes at the die.
- **Inner diameter** is reported at mid-height (`id_change_pct`) and as the minimum over the inner
  surface (`id_min_change_pct`); they differ by a few points when the inner surface barrels.
- **Structured meshes only.** For an unstructured option, triangulate jittered interior points plus the
  boundary points with `scipy.spatial.Delaunay` (the cross-section is convex).
- **Contact events are abrupt.** When a side node is projected onto the die, the force curve shows a
  small jump (visible for m = 1.0 near 21%). Smoother contact or remeshing would remove it.
- **Isothermal, flat dies, 2D axisymmetric**: no temperature coupling, no ovality, no die shapes.
- Mesh is a structured triangulation of the cross-section; elements near the die corners distort first.
- This is a teaching-scale solver, not an industrial forging code; do not compare its speed with one.

## Where to extend next

1. Remeshing (gmsh) + field transfer, so sticking runs reach 50% reduction.
2. Smooth contact (penalty or augmented Lagrangian) to remove force jumps.
3. Thermal coupling (die chilling) — this would also break the K/V invariance and make force-based
   friction identification more interesting.
4. A learned flow-stress law evaluated at quadrature points (the "Surrogate Physics" idea).
