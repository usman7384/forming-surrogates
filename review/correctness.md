# Correctness review (physics, numerics, statistics, code)

Reviewer: independent agent, fresh context, read-only. Scope: whether the code and the results are *correct*.
Tracing numbers to files and checking the rules was done in earlier audits and is not repeated here. All reproductions ran
on main with `.venv` and `FS_DATA_ROOT=~/forming-data` (read only). Scratch scripts were deleted afterwards.
Their key output is quoted below.

## Verdict: **correct with caveats**

I found no bug that changes a headline number by more than its stated precision. The solver's weak form, its
incompressibility treatment, friction law, force definition and contact handling are right. The identifiability
headline reproduces independently to the third digit. GP metrics reproduce from the shipped artifacts. Splits do not
leak. The monitoring statistics match reference formulas. One secondary number (where_to_measure, m unknown,
m_f = 0.5) moves by about 9% once a force-scaling factor the docstring wrongly calls 1 is included. Several modelling
choices and doc claims need a fix or a caveat (list below).

`python -m pytest -q`: **86 passed** (85 s).

---

## 1. Solver (`ringsim/solver.py`)

**Weak form: correct.** Levy-Mises with eta = sigma_bar/(3 q) gives s = 2 eta eps_dot (lines 245-248). The strain
components include the hoop term u_r/r (l.122, l.142). The metric (1,1,1,2) counts rz twice (l.125). Everything is
r-weighted (l.145). Taylor-Hood P2/P1 is inf-sup stable. The divergence block includes the hoop rate (l.152). The V1
patch test (exact to 0.002% force, 0.005% ID) exercises all of this, including the hoop term.

**Friction: correct.** The residual term is c u_r v_r with c = m_f k/sqrt(u_r^2 + delta^2) and k = sigma_bar/sqrt(3) at
the facet quadrature points (l.255-257). This is dissipative and opposes radial slip against a die with no radial
velocity. delta = 0.01 V is small next to the typical v_r of about 0.75 V. V5 (0.43 points from no-slip) is the
expected regularisation gap.

**Newton tangent: consistent except for one term (bug, does not change results).** A finite-difference check of the
full (u, p) Jacobian on an nz = 2 mesh with friction active (82 dofs, random state):

| tangent check | max relative error |
| --- | --- |
| tau_y frozen in the residual (what the tangent assumes) | 4e-10 (m = 0.15), 3e-10 (m = 1.0) |
| actual residual (tau_y depends on u through the facet strain rate) | 1.6e-2 (m = 0.15), 6.2e-2 (m = 1.0) |

The power-law part is exactly consistent. The friction part omits d tau_y/du. The comment at l.262 says so, but the
module docstring (l.9) claims "the consistent tangent ... of the friction law". Effect: none on the results, only on
convergence speed. The converged residual is the exact one, and 0 of the solves in 460 runs failed to converge
(index.csv `newton_unconverged` summed = 0). Fix: correct the l.9 docstring, or add the m_f m K q^(m-1) term.

**Force: correct.** I computed three force estimates at every solve of three 30% runs (nz = 8):

| m_f | power balance vs consistent reaction (sum of residuals over the die u_z dofs) | direct traction 2 pi int (p - 2 eta e_zz) r ds vs power |
| --- | --- | --- |
| 0.0 | 5e-14 | 0.000% |
| 0.3 | 1.4e-10 | -0.5% at start, -2.9% at end |
| 1.0 | 1.8e-10 | -6.7% to -11.0% |

The power balance equals the variationally consistent reaction to round-off, as it must, since u·(Au + B^T p) = -V·sum
R_top and Bu = 0. The direct traction integral is the less accurate estimator: it uses P1 pressure and P2 gradients
evaluated on the boundary, near the singular die-edge corner. So the code uses the right estimator. The 3-11% spread
is still a useful signal that the force carries discretisation error at high friction. This agrees with the README's
"force is less mesh-converged" caveat. At m_f = 0, the final force is 15.09442 against the exact homogeneous value
15.09462 (-0.0013%).

**Contact: correct for these cases.** Contact is never released. I checked the sign of the per-dof die reaction at
every solve for m_f = 0.3, 0.7 and 1.0: no tensile contact dof in any solve, so release is never needed here. Nodes
that land on the die are clipped and join contact once a whole facet is on the die. That is the source of the
disclosed force jumps.

**Volume drift:** -0.08% (m_f 0.3), -0.35% (0.7), -0.53% (1.0) at 30%. The worst over the dataset is 0.96%
(index.csv), against 0.74% in V6, which covers the 15 verify runs only. Fine.

**Friction trend: right sign and shape.** For the 6:3:2 ring at 30% (m = 0.15, sweep in identifiability.json):

- m_f = 0 gives +19.52%, which is exactly the homogeneous 1/sqrt(0.7) - 1.
- ID decreases monotonically through zero at m_f ≈ 0.23.
- The curve flattens towards -22.2% at sticking.

This is the textbook shape. No quantitative comparison with published charts exists (S7 harness empty), and README
§1 says so.

**Questionable: force is not monotone in m_f below the shaded band.** F*(30%) peaks at m_f = 0.425 (1.8283) and falls
to 1.8223 at 0.5. The jumps start only above 0.5. At a fixed configuration the minimum dissipated power cannot
decrease with m_f, so this is a geometry or contact-history effect. It makes the force-only likelihood flat or bimodal
in 0.4-0.5. README/figures shade from 0.55. Suggest starting the shading at about 0.45, or saying force is
non-monotone from 0.43.

**Verify reproduction: not byte-exact (nit, against rule 3's wording).** `python -m ringsim.verify` (76 s) matches
docs/reference/verify_report.md in every row except V2's round-off digits: "1.4e-13 (K x10) and 2.9e-13 (V x10)"
against the reference "6.8e-14 and 0.0e+00". Line endings also differ. These are BLAS/platform round-off, not a
behaviour change. Fix (solver owner): print sub-1e-12 differences as "< 1e-12" so the report is reproducible.

## 2. Data (`common/data.py`, `ringsim/campaign.py`)

**Correct.**

- `Run.at`: `np.interp` on the per-step reduction curve. NaN only beyond the last reached reduction (with 1e-9 slack),
  so early-stopped runs contribute rows only where they got to.
- Reductions are exact multiples of 0.01, so 0.1/0.2/0.3 are grid points and no interpolation error enters.
- Curve lengths agree (stroke = force = id curve = n+1 entries in all exit paths), so the truncation in `load_run` is
  a no-op.
- `scalar_table` groups by run.
- I checked the full dataset: 0 test_random / test_ood_geometry / test_fine_mesh cases equal a train case.
- All 20 test_fine_mesh cases are copies of test_random cases on nz = 12. That is by design (discretisation shift,
  not generalisation) and documented, so it is not a leak into train.
- 105/460 runs stopped early, which matches the README.

## 3. GP (`surrogates/gp_bo/gp.py`, `run.py`, `calibration.py`, `support.py`)

**Correct.**

- NumpyGP: standardisation (ddof 0) is the same as in the fit, the constant mean is added in standardised units, the
  Matern-5/2 formula is the textbook one, and predictive variance = outputscale - v'v + noise. The noise term is
  included, and is 2.3e-6 in standardised units, so negligible.
- Bounds: length scale in [0.02, 1e3] on the unit box, noise ≥ 1e-6. Sensible.
- I recomputed the metrics from `artifacts/gp/id_gp.npz` on the full data. The artifact holds 1040 rows whose targets
  are identical to scalar_table(train). RMSE 0.0930 / 0.8054 / 0.1421 and coverage95 0.983 / 0.991 / 0.943 match the
  README.
- Conformal factor: k = ceil((n+1)·0.95) on a run-wise 20% hold-out of train. This is the correct split-conformal
  quantile. The test splits are not used in it.

Caveats:

- **Questionable** (`run.py:290-292`): the choice between the conformal and the raw sd is made from *test_random and
  test_ood* coverage, so test data drives a deployment decision. It changes no number (with factor 0.91, coverage is
  unchanged at 3 d.p.), but the docstring's "test splits not used" is only true of the factor. Fix: decide the rule on
  the calibration hold-out.
- Coverage counts rows that are correlated within a run (3 per run). This is disclosed in calibration.py; fine.
- Support flag (`support.py:52-55`): the logic matches its documented rule. Fine.

## 4. Identifiability (`identifiability.py`, `measurement_design.py`)

**Headline reproduced independently.** My script marginalises log K analytically (Gaussian integral) instead of using
a grid, integrates over a 20001-point m_f grid with the trapezoid rule, reads the 26 cached sweep runs directly, and
replays the same RNG sequence:

| | file | independent |
| --- | --- | --- |
| force only, one draw | 0.1034 | 0.1034 |
| force + ID, one draw | 0.00459 | 0.00459 |
| force + K known, one draw | 0.1003 | 0.1003 |
| means over 100 draws (f / id / k) | 0.1047 / 0.00460 / 0.0698 | 0.1047 / 0.00460 / 0.0699 |
| ratio over draws | 22.75 | 22.75 |

- The likelihood, the priors (uniform m_f, log K ~ N(0, 0.075²)) and the normalisation are right.
- The 0.0046 is essentially sigma_ID/|dID/dm_f| = 0.3/65.25 = 0.00460: force adds almost nothing once ID is
  measured. That is honest, and worth one sentence in the docs.
- Held-out node: dropping m_f = 0.25 from the sweep gives a PCHIP error at 0.25 of -0.002 ID points and 0.013% force.
  Inverting the *real* solver run at 0.25 with the 25-node model gives mean 0.25008, std 0.00461 (bias 0.02 sd). So
  the inverse crime of interpolation is not inflating the result.
- Sensitivity: if the force model uses all 70 cached m = 0.15 runs (including the inverse-study nodes) instead of the
  26 sweep nodes, the single-draw force+K-known std moves from 0.100 to 0.093. Force-only conclusions depend on the
  roughness of the force curve. This is consistent with the stated caveats.

`measurement_design.py`:

- Grid refinement (m 11 to 101 points, m_f 201 to 1001) changes no quoted number by more than 0.0002 (for example,
  0.00633 to 0.00633, 0.02321 to 0.02311).
- **Bug in the docstring/notes, changes one number slightly:** l.14-15 and the notes say "V/h0 = 1 in the dataset
  normalisation". For the standard ring V/h0 = 1/(2/3) = 1.5 (`solver.py:193,201`), so F* has 1.5^m divided out. With
  m unknown, the physical force carries an extra m-dependent amplitude (2% to 13% over the m range). Putting it back:

  | m_f | force only, m unknown | ID@30, m unknown |
  | --- | --- | --- |
  | 0.10 | 0.1395 to 0.1427 | unchanged |
  | 0.25 | 0.1311 to 0.1403 | unchanged (0.0063) |
  | 0.50 | 0.1249 to 0.1238 | **0.0232 to 0.0212** |

  The README quotes 0.0232. Fix: either include log(V/h0)·m in LF, or correct the docstring and say that 0.0232 is
  the number without that factor.

## 5. Design (`design.py`)

**Correct.**

- The objective is the mean over truths of a noise-free grid-posterior std, with the GP variance added to the
  measurement variance.
- Common range: both 11-node sweeps completed to 30%, so the common range is [0, 1] with no extrapolation. Factor 1.70
  confirmed.
- With 26 nodes instead of 11, the standard-ring solver objective goes from 0.015729 to 0.015725, so PCHIP node
  spacing is irrelevant.

**Questionable:** per-truth std rises from 0.003 (m_f 0.05) to 0.046 (m_f 0.85), so the objective is dominated by
m_f > 0.6. That is where ID flattens and where thin rings fold over (censored data). The "best ring" is therefore
mostly "best at high friction". Say so next to the 1.64/1.70 factors.

## 6. GNN (`surrogates/gnn/*`, `baselines/pod_field.py`)

**Correct.**

- The normaliser is fitted on the train runs only, including the extra mixmesh train runs. Validation runs are drawn
  from `train` by run.
- Edge features are initial-mesh relative (dr, dz, |d|).
- ID probe: the (r_min, z = 0) node of the initial mesh. On all 40 full test_random runs, the probe ID equals the
  solver's `id_change_pct` curve at every saved step to 9e-6 points (float32).
- J = (A/A0)(rbar/rbar0) is the exact axisymmetric volume ratio of a linear triangle.
- nRMSE = RMSE / std(truth over the split). The POD baseline uses the same `_summarise`, so the comparison within a
  split is like for like. Do not compare nRMSE across splits: the denominators differ.
- POD resampling is done in initial normalised coordinates, and the resampling floor is reported.
- Mixmesh leakage: fine_train has its own LHS seed (66 runs, nz 12), 0 exact overlaps with any main test split, and
  nearest normalised distance 0.54.

Nits:

- `evaluate.py:128`: the predicted mesh's J uses the *predicted* step-0 positions as reference rather than p0, unlike
  the training loss. This only matters if the model's s = 0 displacement is nonzero.
- README "training on 60 extra runs": with `extra_val_frac` 0.15, 51 are trained on and 9 are used for validation.

## 7. Plant (`plant/press_sim.py`, `serving/edge/*`)

**Correct.**

- Wilson matches scipy's `binomtest(...).proportion_ci(method='wilson')` to 5e-6. The residual difference is from z =
  1.96 against the exact quantile.
- PSI matches an independent quantile-bin implementation exactly. KS is scipy.
- PSI null at 100 vs 50 samples: mean 0.126, P(PSI > 0.25) = 9%. The KS AND-gate is what keeps false alarms down.
  That is consistent with the docstring.
- 3-point Gauss-Hermite nodes and weights (0, ±sqrt3; 2/3, 1/6, 1/6) are correct, and the law of total variance in
  `predict` is right.
- Re-zeroing subtracts mean(recent raw zero) - median(reference zero) before touch-off. It is gated on the primary
  alert, and the monitor sees the raw zero. Correct.
- Labels are applied when due_index ≤ parts processed, which makes the delay deterministic. Latency is
  verdict wall time minus server emit time.

Caveats:

- **Questionable** (`serving/model_api.py:67-73`, lead-owned): `GridFrictionEstimator` adds the GP sd to every label
  as if GP errors were independent between labels. For one lot (fixed od, h, m; reductions within ±0.01) the GP error
  is nearly common to all labels, so mf_std shrinks like 1/sqrt(n) below the GP floor. In-distribution the effect is
  small: GP sd 0.15 points gives a floor of about 0.0023, against the observed 0.0025. On OOD lots (GP sd ~1.1 points)
  the friction std is about sqrt(9) = 3 times too small. This touches `go`/`check` counts, not acceptance (which uses
  the mean). Fix: add the GP variance once per lot, or document the assumption.
- **Nit** (`service.py:42`): RED_MAX = 0.35, "the reduction range the surrogate was trained on", but the GP was trained
  on reductions 0.1-0.3 (gp.py LO/HI). Queries in (0.30, 0.35] extrapolate without the flag. In the shipped drift run
  this affects 9 parts, all already `check`, so no number changes. Fix: RED_MAX = 0.30.
- **Nit** (`align.py:77`): end_pos = max of a noisy position. That biases reduction by about +0.001 (0.01 mm over
  ~10 mm), which is negligible.

## 8. Tests

**Guard correctness:**

- `test_frictionless_matches_exact_solution`, `test_shape_is_invariant_to_K_and_speed`, `test_more_friction_closes_the_ring`
- `test_numpy_posterior_matches_gpytorch`, `test_interpolates_training_points`
- `test_grid_posterior_recovers_truth_and_id_collapses_ridge`, `test_discrepancy_study_bias_signs_on_synthetic_model`
- `test_conformal_scale_fixes_overdispersed_sd`, `test_crps_is_minimised...`, `test_split_runs_is_by_run...`, `test_split_by_run_no_leakage`
- `test_targets_use_initial_mesh_and_probe_matches_curve`, `test_j_on_solver_data_is_near_one...`
- `test_wilson_known_values`, `test_psi_and_ks_detect_a_shift...`
- `test_features_recover_reduction_within_noise`, `test_verdict_logic_go_check_stop`, `test_rezero_recovers...`, `test_each_drift_touches_only_its_own_channel`

**Smoke or plumbing only:** train/resume/DDP, evaluate-and-gallery, dashboard, OPC-UA end-to-end, bundle rerun,
harness/aggregate on smoke, spec and config hash.

**Missing (recommended):**

1. A finite-difference Jacobian test on an nz = 2 mesh (it would have flagged the friction-tangent docstring).
2. A force = consistent-reaction test.
3. A golden-value regression of one default solver run (ID and F* at 30%) independent of verify.
4. An analytic-K posterior check of `posterior()` against a closed form.
5. A test that `GridFrictionEstimator` matches a hand-computed Gaussian posterior.

## Fix list (by owner)

1. solver-engineer: `ringsim/solver.py:9`, docstring "consistent tangent ... friction law". Either correct the wording
   or add the d tau_y/du term. **Not result-changing.**
2. solver-engineer: `ringsim/verify.py`, V2 round-off formatting ("< 1e-12"), so the report is reproducible across
   machines.
3. gp-scientist: `surrogates/gp_bo/measurement_design.py:14-15` and notes. V/h0 is 1.5, not 1. Include the factor
   (0.0232 becomes 0.0212 at m_f 0.5, m unknown) or restate the assumption. Then README §2 bullet.
4. gp-scientist: `surrogates/gp_bo/run.py:290`. Pick the served sd on the calibration hold-out, not the test splits.
5. gp-scientist / lead: force non-monotone from m_f ≈ 0.43. Widen the shaded band or add a sentence.
6. gp-scientist: design.py/README. State that the design objective is dominated by m_f > 0.6.
7. lead: `serving/model_api.py:67-73`. Correlated GP error across labels of one lot; fix or document.
8. plant-engineer: `serving/edge/service.py:42`, RED_MAX should be 0.30 (the GP training maximum).
9. gnn-engineer / lead: README "60 extra runs" is 51 trained + 9 validation. `evaluate.py:128` J reference (nit).
