# The solver, explained (for an ML engineer new to CAE)

This note explains `ringsim/solver.py`: what equations it solves, how it solves them, how the die
force is computed, and the three things that go wrong in a simulation like this (contact, fold-over,
volume drift). You should be able to follow it without reading the code. Code names are given so you
can find things later.

Honesty note: this solver is **verified** (it solves its own equations correctly, checked against
exact solutions and convergence tests) but not **validated** (nothing here is compared with a real
experiment). It is teaching-scale, written with AI assistance, and makes no speed claim against
industrial solvers. Every number below is read from `results/verify/report.md` (default checks) or
`results/solver/verify_extended/report.md` (command: `python -m ringsim.verify --out results/solver/verify_extended --extended`) or from
`results/solver/s4_unstructured_vs_structured.json`.

## 1. The physical problem in one paragraph

A ring (think: a thick washer) is squeezed between two flat rigid dies. Metal flows sideways like a very
thick fluid; the ring gets shorter and wider, and the **friction** between ring and die decides whether the
hole grows (low friction) or shrinks (high friction). That is why the hole size is a friction gauge. The
ring is axisymmetric (the same all the way round), so we only simulate its 2D cross-section in
coordinates (r, z), and by symmetry only the top half (z >= 0). The top die moves down at speed V.

## 2. From physics to a weak form

**Material.** Hot metal under fast deformation behaves like a viscous fluid whose resistance rises with
strain rate: flow stress `sigma_bar = K * epsdot_eq^m` (Norton-Hoff law; K is strength, m the rate
sensitivity, 0.05-0.3 here). Elastic effects are ignored (rigid-viscoplastic), so the unknown is the
**velocity field** u(r, z), not displacement. `epsdot` is the strain-rate tensor, `epsdot_eq =
sqrt(2/3 epsdot:epsdot)` its scalar measure. The Levy-Mises flow rule says the deviatoric stress is
proportional to strain rate, `s = 2 eta epsdot`, with an **effective viscosity** `eta = sigma_bar /
(3 epsdot_eq)` (code: `eta` in `operators`). Because eta depends on u, the problem is nonlinear.

**Incompressibility.** Metal volume does not change: `div u = 0`. The pressure p is the Lagrange
multiplier that enforces it. Total stress is `sigma = s - p I`.

**Balance of momentum** (no inertia, no gravity): `div sigma = 0` in the ring.

**Boundary conditions.**
- Mid-plane z = 0: symmetry, `u_z = 0` and no shear.
- Top face under the die: `u_z = -V` (the die pushes), and a friction traction in r: Tresca friction says
  the shear stress opposes sliding and is capped at `m_f * sigma_bar / sqrt(3)` (m_f in [0, 1] is the
  "friction factor"). A hard switch at zero sliding speed is not differentiable, so it is smoothed:
  `tau = -m_f (sigma_bar/sqrt3) u_r / sqrt(u_r^2 + delta^2)` (`delta = delta_rel * V`).
- Free surfaces (inner and outer wall): traction-free.

**Weak form, step by step.** A *weak form* is the strong equations multiplied by arbitrary "test
functions" and integrated, so that derivatives move onto the test function and the unknown only needs one
derivative. Take a test velocity w (zero where u is prescribed) and a test pressure q.

1. Multiply `div sigma = 0` by w and integrate over the 3D ring. A volume element of the revolved body
   is `r dr dz dtheta`, so a 3D integral becomes `2 pi * integral(... r dA)` over the 2D cut `A`. The
   factor `2 pi` drops out of the equations; the **weight r survives in every integral** because a point
   at radius r represents a circle of circumference 2 pi r.
2. Integrate by parts (divergence theorem): `-integral(div sigma . w r dA) = integral(sigma : eps(w) r dA)
   - boundary integral(traction . w r ds)`. Free surfaces and the symmetry plane contribute nothing. On the
   die face the test function has `w_z = 0` (u_z is prescribed), so only the friction traction remains.
3. Insert `sigma = 2 eta eps(u) - p I`. The axisymmetric strain-rate components are
   `eps_rr = du_r/dr, eps_zz = du_z/dz, eps_tt = u_r/r, eps_rz = (du_r/dz + du_z/dr)/2`
   (`_eps` in the code). The hoop term `u_r/r` appears because a point moving outward stretches its
   circle. So `eps:eps' = rr rr' + zz zz' + tt tt' + 2 rz rz'` (the factor 2: the metric `_G`), and
   `div u = eps_rr + eps_zz + eps_tt`.

Result: find u (with the boundary values) and p such that for all w, q:

```
 integral 2 eta(u) eps(u):eps(w) r dA  -  integral p div(w) r dA  +  integral_die c(u) u_r w_r r ds = 0
 integral q div(u) r dA = 0
 with  c(u) = m_f sigma_bar / (sqrt3 * sqrt(u_r^2 + delta^2))      (friction as a nonlinear "dashpot")
```

Read it as: the viscous resistance, plus pressure, plus die friction, must balance for every possible
virtual motion w; and the flow must be volume-preserving.

## 3. Finite elements: turning it into a matrix problem

The cross-section is cut into triangles. Velocity is quadratic (P2) in each triangle, pressure linear
(P1) - the **Taylor-Hood** pair. The pairing matters: with equal orders the discrete pressure can
oscillate (violating the "inf-sup" stability condition); Taylor-Hood is a standard stable choice.
Writing u and p as vectors of nodal values gives a saddle-point system

```
 [ A(u)  B^T ] [u]   [0]
 [ B     0   ] [p] = [0]        residual  R(u, p) = [ A(u) u + B^T p ;  B u ]
```

`A(u)` is the viscous + friction matrix evaluated at the current strain rates (a "secant" matrix: A(u) u =
internal forces), `B` is the weighted divergence. `_Geometry` precomputes the strain of every basis
function at every quadrature point and assembles with `einsum` (vectorised, instead of looping over basis pairs in Python; speed-up not measured here).
Prescribed velocities (die speed, symmetry plane) are removed from the unknowns by `condense`.

Mesh: by default a structured grid (each rectangle cut into two triangles), about `nz = 8` elements
through the half height. There is also an **unstructured option**, see section 9.

## 4. Newton's method and the tangent

`R(u, p) = 0` is nonlinear because eta (and the friction coefficient c) depend on u. **Newton's method**
linearises R around the current guess and solves for a correction: `J(u) [du; dp] = -R`, `u += du`. The
matrix J is the **tangent** (Jacobian). A plain fixed-point iteration (freeze eta, solve, repeat) would
need hundreds of iterations for small m, where the material is nearly rigid-plastic and eta varies
violently with strain rate.

*Deriving the tangent of the power law.* Let `q = sqrt(2/3 eps:eps + floor^2)` (the floor `eps_reg * V/h0`
keeps eta finite where the strain rate is zero). With `eta = K q^(m-1)/3`:
`d eta / d q = (m-1) eta / q` and `d q = (2/3)(eps:d eps)/q`. The derivative of the internal force
`integral 2 eta eps(u):eps(w) r dA` in the direction `du` is therefore

```
 integral 2 eta eps(du):eps(w) r dA  +  integral (4/3)(m-1) eta / q^2 * (eps:eps(du)) (eps:eps(w)) r dA
```

The first term is the secant matrix `A`; the second is a rank-one correction per quadrature point
(`geo.tangent` in the code). For m < 1 it is negative: stiffness is lower than the secant suggests, and
Newton steps are longer than the fixed-point ones. For m = 1 (Newtonian) the second term vanishes and
Newton is exact in one step.

*Tangent of the friction.* `d/du [u/sqrt(u^2+delta^2)] = delta^2 / (u^2+delta^2)^(3/2)`, so friction adds
`tau_y delta^2 / s^3` on the die face (`s = sqrt(u^2+delta^2)`). Friction is *stiff* in a narrow band
around zero sliding speed (width delta) - that is the numerical price of the smoothing. Smaller delta
mimics true Tresca better but is harder for Newton (section 8).

*Honest caveat.* Inside the friction tangent the flow stress sigma_bar is frozen (its dependence on u is
dropped), so the tangent is "consistent" for the power law but only approximate for friction. Newton then
converges superlinearly rather than quadratically; the maximum over all solves was 25 iterations in the
default verification and 40 in the delta_rel = 1e-3 run (V6, V5b; typical counts are not reported in a results file).

*Globalisation.* If the residual norm grows after a step, the previous step is backtracked (halved, up to
6 times). The stopping rule is `||du|| / ||u|| < 1e-8`. Every solve that fails to reach it is counted
(`newton_unconverged_solves`) and shows up in verification check V6 (0 failures in all reported runs).

## 5. The die force from a power balance

We need the force F on the die (the "force-stroke curve"). Reading it off stresses at the die face is
noisy: stress is a derivative of u, so it is less accurate than u itself, and it is evaluated at an
edge with a steep gradient. Instead use **energy conservation**.

Put the *solution itself* in as the test function, `w = u` (allowed once you account for the die term):
the pressure term disappears because `integral p div u r dA = 0` (incompressibility; discretely it is
exactly `p . (B u) = 0`), and what is left is: the power delivered by the die equals the power dissipated
in the metal plus the power dissipated by friction sliding:

```
 F * V = 2 pi * ( integral 2 eta eps:eps r dA  +  integral c u_r^2 r ds )
```

(`2 pi` restored for the full revolution; the half model has one die). In the code: `power = u @ (A @ u)`
and `force = 2 pi * power / V`. Because it is an integral of u rather than of its derivatives, it is
expected to be smoother than a direct stress integral (not compared here). Check V1 shows only that, with zero
friction and a homogeneous field, the force is within 0.002% of the exact formula `K (V/h)^m pi (Ro^2 - Ri^2)`.

## 6. Moving the material: the time loop

Rigid-viscoplastic flow has no memory except geometry and accumulated strain, so we repeat: solve for u on
the current shape; move the nodes; accumulate strain; repeat (**updated Lagrangian**: the mesh follows
the material).

- The die goes down by `dh = 1%` of the initial half height per step.
- **Midpoint rule** (second-order in time): move nodes half a step with u, re-solve on that mid-shape,
  then move the full step with the mid-step velocity. Cost: two Newton solves per step.
- Strain per element grows by `dt * strain rate` (used by hardening and stored in the dataset).
- Fields are saved every `save_every = 2` steps.

## 7. Contact, fold-over and volume drift

**Contact.** The die is a straight line at height z_die. Boundary nodes sitting on it (within a tiny
tolerance) get the die velocity and friction. Metal bulges outward; a free-surface node that would
cross z_die is **clipped** back onto the die (`np.clip(p[1], 0, z_die - dh)`) and from then on belongs to
the contact set. Contact nodes are never released. That is a simplification: real contact is
unilateral (a node leaves the die when the pull-off force changes sign). For a squeeze with no unloading
it is adequate; it would be wrong for a die that retracts. A visible consequence is the **step in the m_f = 1 force curve near
22% reduction** in `results/verify/calibration_chart.png` (annotated on the figure): in one step an outer
side-wall node is clipped onto the die and joins the contact set (an informal check: I re-ran the V3 m_f = 1 case
with fields saved every step and the die-node count rises by one at that step; not stored in a results file), and the force jumps. It is an artefact of node-wise contact
without smoothing, not physics. The no-slip curve shows no such step before it stops.

**Fold-over.** With high friction the outer wall rolls over onto the die (a real phenomenon), so the
triangles on that wall are squeezed flat and eventually flip. A triangle whose signed area changes sign,
or falls below `min_area_frac = 5%` of the mean, is called collapsed. We have no remeshing (re-triangulating
the distorted body), so the run **stops cleanly**, saves the state, and sets `stopped_early` with a
warning. These runs are kept in the dataset and flagged: in the default verification `fric_0.7`, `fric_1.0`
and `noslip` stop before 40% reduction; 8 of the 40 unstructured runs stopped early. Because the
final reduction then differs per run, ML targets must be read from the curves at fixed reductions
(`common.data.scalar_table`), never from the final-state summary.

**Volume drift.** Metal is incompressible, so the volume of the deformed mesh should stay constant.
It does not exactly, for three reasons: (1) `div u = 0` is only enforced weakly, in the sense of P1 test
functions; (2) the mesh moves by its corner velocities and drops the curvature of the quadratic
(mid-edge) velocity, so the geometry is a polygon approximation; (3) clipping at the die removes a little
volume. We measure it with Pappus' theorem (volume = 2 pi x centroid radius x area, exact for
triangles). Typical size: 0.0022% for the frictionless homogeneous-field test (V1) and a worst case of 0.74% across the
15 default verification runs (0.89% across the 40 unstructured dataset runs). It is always slightly
negative and grows with friction. It is a **spatial** error: refining the mesh reduces it; shrinking the
time step does not.

## 8. What verification shows, including the weak spots

Run: `python -m ringsim.verify --out results/verify` (default checks, reproduces
`docs/reference/verify_report.md` apart from the round-off text in V2) or add `--extended` for the
checks below. Default numbers in `results/verify/report.md`, extended ones in `results/solver/verify_extended/report.md`.

| Check | Result |
| --- | --- |
| V1 frictionless homogeneous-field test (40%); formerly called a "patch test" | force 0.002%, inner diameter 0.005%, strain 0.001% from the exact solution. The exact solution is a homogeneous (linear) field that the P2/P1 space contains, so this only shows the discretisation can reproduce an exact homogeneous solution, and that assembly, the power-balance force and the time loop are consistent for it. It would still pass with a wrong friction law or wrong contact handling: it is **not** evidence about friction, contact or non-homogeneous flow. The default report text still says "Patch test" (frozen by rule 3); the extended report carries a V1-label row with this caveat |
| V1b same homogeneous-field test on 3 random unstructured meshes | worst of three seeds: 0.002% / 0.005% / 0.001%, volume drift 0.0022% - the same as the structured mesh (the exact solution is representable on any mesh, so the numbers match to the reported digits) |
| V4 mesh convergence at m_f = 0.3 (30%) | inner diameter -4.42 / -4.44 / -4.44% for nz = 6 / 8 / 12 |
| V4b mesh convergence at m_f = 0.7 (25%) | inner diameter -15.39 / -15.34 / -15.39%: converged to within 0.05 points, but **not monotone** (nz = 6 is as close to nz = 12 as nz = 8), so no convergence order can be claimed from three meshes. Normalised force 1.6599 / 1.6505 / 1.6369: nz = 8 is still 0.83% above nz = 12 - the **force is less mesh-converged than the inner diameter** at high friction. The gating criterion is now "nz=8 within 0.25 points of nz=12 on the inner diameter" (passes at 0.04); the force gap is reported as a separate non-gating INFO line (V4b-F). History: the first run used V4's monotone rule and failed only because both errors were below 0.05 points; I replaced it with an absolute 1-point tolerance after seeing that result, and on reviewer advice tightened it to 0.25 points. The numbers did not change between versions. |
| V5 sticking bound (29%) | Tresca m_f = 1: -21.19%; exact no-slip: -21.62%; gap 0.43 points |
| V5b smoothing sensitivity | delta_rel 1e-2 -> 1e-3 moves Tresca m_f = 1 from -21.19% to -21.55% (no-slip -21.62%): gap 0.43 -> 0.07 points. So **about 0.4 points of inner-diameter change at high friction is an artefact of the friction smoothing** - small compared with the 40+ point range of the calibration chart, but not small compared with a measurement uncertainty of a fraction of a percent. The price: Newton's maximum iterations rise from 25 to 40 |
| V6 / V6b health | no Newton solve missed 1e-8; worst volume drift 0.74% (default runs), 0.58% (extended runs) |

## 8b. Further checks added after the expert review

All are in `results/solver/` and written by scripts (commands inside each JSON or below).

**Rate exponent (V3b, `--extended`; `results/solver/verify_extended/rate_sensitivity.json`).** The inner-diameter
change at 30% reduction for the standard ring, m_f = 0 ... 1, at rate exponents m = 0.05 / 0.15 / 0.30. At
m_f = 0 all three agree (+19.5%, the exact homogeneous value). With friction the curve shifts: at m_f = 1 the
inner-diameter change is -24.4 / -22.2 / -20.1%; the largest shift against m = 0.15 is 5.40 points (m = 0.05) and
3.89 points (m = 0.30), at high friction; at m_f <= 0.1 the shift is under 0.6 points. The total span over
m_f is 43.9 / 41.8 / 39.6 points. Meaning: independence from the strength K is exact by construction (V2), but
independence from the **rate exponent is not**: the claim "the inner diameter barely depends on strength or
rate" holds for K and for the die speed V, not for m. A friction estimate from an inner-diameter measurement
therefore needs the rate exponent of the material (or must carry its uncertainty), most strongly at high friction.

**Published calibration charts (harness only).** `results/solver/published_ring_calibration.yaml` is an empty
table with a documented row schema, and `python -m ringsim.calibration_compare` compares ringsim at each row's
geometry, friction and reduction (output `published_ring_comparison.json`). Status: **harness built, no
verified source data supplied**; no values were digitised or typed from memory. No new analytic reference was
added: the only closed form I derived and used is the frictionless homogeneous compression (V1), which is not
a friction reference; I know of no closed form for sticking friction that I could verify myself.

**Fold-over censoring (`results/solver/censoring.json`; `python -m ringsim.dataset_facts`).** In the full dataset
(460 ok runs, all splits), 105 (22.8%) stopped before 35% reduction. The fraction of runs in each friction
bin that reached a given reduction is in the file; for example at 35%: 100% for m_f < 0.4, 83% for
0.4-0.6, 54% for 0.6-0.8 and 53% for 0.8-1.0 (see the file for all bins and reductions). All bins reach 15% in
every run. Runs stopped early are right-censored: a surrogate has no data beyond their last step, so
predictions at high friction and high reduction extrapolate from the surviving, selected runs.

**Unstructured dataset facts (`results/solver/unstructured_dataset_facts.json`).** 40 planned, 40 ok, 8 stopped
early, worst volume drift 0.89% (the numbers quoted in section 7 and 9).

## 9. The unstructured mesh option (task S1) and why it exists

The default mesh has all element edges aligned with the r and z directions. A surrogate (especially a
mesh GNN) trained on such meshes might learn properties of the mesh rather than of the physics. The option
`Numerics(mesh="unstructured", mesh_seed=k)` builds a jittered point lattice (interior points move
by up to 25% of the spacing in both directions; boundary points only slide along their edge; corners are
fixed) and triangulates it with Delaunay (`make_mesh`). The ring cross-section is a rectangle - a convex
domain - so every boundary edge is a Delaunay edge and the die, symmetry and wall conditions work
unchanged. The default stays `"structured"` and the default numerics stored in each run file are
unchanged (the new keys are written only for unstructured runs). The dataset
`$FS_DATA_ROOT/unstructured` (40 runs, split `test_unstructured`, 40 points inside the training box with
a separate Latin-hypercube seed, a different mesh seed per run) is made with
`python -m ringsim.campaign --out $FS_DATA_ROOT/unstructured --mesh unstructured --n-unstructured 40`.
The comparison with the same 40 cases on the structured mesh is in
`results/solver/s4_unstructured_vs_structured.json` (command inside the file). Result (unstructured minus
structured, same nz = 8): at 10% reduction the inner diameter differs by 0.03 points RMS (max 0.07) and the
normalised force by 0.25% RMS (max 0.71%); at 20%, 0.09 points RMS (max 0.28) and 1.19% RMS (max 3.7%); at
30% (33 of 40 pairs reached it), 0.20 points RMS (max 0.69) and 2.5% RMS (max 9.6%). Reading: the inner
diameter, the quantity the identifiability study relies on, is almost mesh-independent; the **force
is not**: mesh sensitivity reaches a few percent at 30% reduction (qualitatively consistent with V4b), so
force-based conclusions should be stated with that caveat (I did not compare it with the force spread
across friction values). Eight of the 40 unstructured runs and seven of their structured twins stopped early by fold-over,
so which cases fold is itself mesh-dependent.

## 10. What can go wrong (and what to check)

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `stopped_early = true` | fold-over at high friction / thin rings | Expected; use curves at fixed reductions; do not trust final-state scalars |
| `newton_unconverged_solves > 0` | very small m or delta, bad mid-step guess | Check `warnings`; reduce `dh_frac`; do not use the run unflagged |
| Volume drift above ~1% | coarse mesh or heavy clipping | Refine `nz`; compare with `nz` x 1.5 |
| Force differs between meshes by ~1% at high friction | force not mesh-converged (V4b) | Treat force-curve differences below ~1% as discretisation noise |
| Inner diameter at high friction off by ~0.4 points | friction smoothing (V5b) | Use `delta_rel=1e-3` if sticking accuracy matters (costs more Newton iterations) |
| `MemoryError` in a campaign | too many workers / other jobs on the box | Rerun the same command: finished runs are skipped |

## 11. How to run

```
python -m pytest -q                                              # fast tests
python -m ringsim.verify --out results/verify                   # default checks (~3 min)
python -m ringsim.verify --out results/solver/verify_extended --extended   # + V1b, V4b, V5b, V6b
python -m ringsim.campaign --out $FS_DATA_ROOT/unstructured --mesh unstructured --n-unstructured 40
python -m ringsim.dataset_facts --out results/solver        # censoring.json, unstructured facts
python -m ringsim.calibration_compare --out results/solver  # empty harness
python -m ringsim.mesh_compare --unstructured $FS_DATA_ROOT/unstructured --twin $FS_DATA_ROOT/unstructured_twin --out results/solver
```
