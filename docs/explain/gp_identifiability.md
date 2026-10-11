# Explainer: GP surrogate, identifiability and design (area `gp_bo`)

For an ML engineer who is new to CAE. Numbers are in `results/gp/`, `results/identifiability/`,
`results/design/`; this page explains the choices, not the results. Status: Wave 2, run on the `full` dataset (460 runs).

## What was built
| Piece | File | Command |
| --- | --- | --- |
| Scalar GP + baselines (G1) | `surrogates/gp_bo/run.py`, `gp.py`, `surrogates/baselines/scalar_*.py` | `python -m surrogates.gp_bo.run --data full --out results/gp` |
| K-invariance demo (G2) | `invariance.py` | `python -m surrogates.gp_bo.invariance --data full --out results/gp` |
| Inverse friction by root-finding (G3) | `inverse.py` | `python -m surrogates.gp_bo.inverse --out results/gp` |
| Identifiability posterior (G4) | `identifiability.py` | `python -m surrogates.gp_bo.identifiability --out results/identifiability` |
| Best ring by Bayesian optimisation (G5) | `design.py` | `python -m surrogates.gp_bo.design --out results/design` |
| Serving API (G6) | `api.py` | `python -m serving.check_model surrogates.gp_bo.api:load artifacts/gp` |

## The GP surrogate (G1)
- **What it learns:** inputs (OD/ID, H/ID, m_f, m, reduction) to two targets, inner-diameter change in % and
  log of the normalised force F*. One GP per target.
- **Why fixed reductions:** a run's QoIs in `index.csv` are at its *final* reduction, which differs for runs
  that fold over early. `scalar_table` reads the per-step curves at 0.1, 0.2, 0.3, so every row means the same
  thing. Rows are NaN-dropped where a run did not get that far.
- **Why split by run:** the three rows of one run are nearly the same point. Random row splits would leak.
  The model is fit on `train` only; `test_random`, `test_ood_geometry` (new ring shapes) and
  `test_fine_mesh` (a finer mesh, a discretisation check, not generalisation) are held out whole.
- **Why a GP, why Matern 5/2 with ARD:** the data are few, smooth, noise-free simulator outputs, which is where
  exact GPs shine, and they give an honest uncertainty. Matern 5/2 is twice differentiable, a safe default for
  physical responses. ARD gives one length scale per input; a short one means the target changes quickly with
  that input (influential), a very long one means irrelevant. Inputs are scaled to a fixed box so the length
  scales are comparable.
- **Log force:** force is positive and multiplicative in K and speed, so the log makes it additive.
- **Noise floor:** the simulator is deterministic; noise is bounded below at 1e-6 only to keep the matrix
  factorisation stable.
- **Training vs serving:** GPyTorch fits the hyper-parameters; the saved artifact is just training inputs,
  targets and hyper-parameters (`artifacts/gp/*.npz`). `NumpyGP` rebuilds the Cholesky factor at load time and
  predicts with numpy only (a test compares it with GPyTorch). The plant therefore does not need torch.
- **Reproducibility, honestly:** GP fits are not bit-reproducible across torch thread counts (the floating-point
  summation order changes and the optimiser lands at a slightly different optimum); RMSE and length scales
  differ by a few percent between fits, and the conclusions do not change. `fit_gp` therefore pins the thread
  count (default 4), the seed and deterministic algorithms, and `artifacts/gp/meta.json` records torch, gpytorch,
  numpy and scipy versions and the thread count; a test checks that two fits agree to 1e-8. Determinism holds for
  one thread count, platform and package set, not across them. The saved artifacts are the reference.
- **Calibration (uncertainty you can trust):** coverage alone can hide an sd that is simply too wide, and the three
  rows of a run are not independent trials. So `gp_accuracy.json` also reports CRPS (a score that rewards both
  sharpness and calibration), PIT histograms (flat if calibrated), and a split-conformal scale factor for the sd.
  The factor is learned on a run-wise 20% hold-out of the TRAIN runs (a GP fitted on the other 80%), never on
  test runs, then applied to the GP fitted on all train runs. The plant sees the conformal sd if the data pass a
  fixed rule (both test coverages at least 0.90 after scaling), else the raw sd; `served_sd` in `meta.json` and in
  `gp_accuracy.json` says which. Note: that rule looks at the test coverages, so the choice between the conformal and the raw sd was made with a look at test data (the factor itself is learned on the train hold-out only); the effect is small (the factor is 0.91). The same file lists the MLP ensemble (from `results/gnn/n5_mlp_baseline_full.json`,
  same rows) next to the GP, polynomial and trees. A scale factor can fix an sd that is too wide everywhere; it
  cannot fix an sd that is too wide in the middle but too narrow in the tails (the PIT shape).
- **Data-support flag:** early fold-over censors the training data at high friction and thin rings at high
  reduction (the surviving runs are not representative). `surrogates.gp_bo.api.in_data_support(X)` is False for a
  query when more than half of at least 5 nearby training runs (within 0.10 in m_f and H/ID) ended before the
  requested reduction. It is not part of the frozen `Surrogate` contract. The GP's own sd does not know about
  censoring, which is why a separate flag exists.
- **Baselines:** cubic ridge polynomial and gradient-boosted trees, to show what the GP is worth. Trees cannot
  extrapolate, so the OOD split is where they should struggle.
- **Reading coverage:** a 95% interval should contain about 95% of the test points. Rows from runs that folded over early only exist for the reductions they reached; `early_stop_handling` in
  `results/gp/gp_accuracy.json` reports metrics separately for those runs.

## Invariance (G2)
In this model the shape does not depend on K or die speed; force scales as K V^m. The demo re-solves
dataset cases at K = 0.6 and 1.6 and shows the inner diameter is identical to round-off, then fits a GP with
log K as a sixth input. The ARD length scale of K grows large for the diameter (tens of times the others: irrelevant) and
stays finite for raw log force (relevant, linear in log K). This is also what justifies treating K as an
analytic axis in G4.

## Inverse friction (G3)
"Which m_f gives this inner diameter?" has one unknown, and the diameter falls monotonically with it, so it
is a bracketed root: `scipy.optimize.brentq` between 0 and 1 on the solver, a handful of solver calls. Bayesian
optimisation would be the wrong tool (there is nothing to optimise, only a root to find). The same root on the
GP mean costs no solver calls and is reported with its error.

## Identifiability (G4)
- **Idea:** the force is K times a solver curve; the diameter is a solver curve independent of K. After one
  sweep of the solver over m_f (26 runs), the probability of every (m_f, K) pair given the data is a few
  lines of arithmetic on a grid (exact Bayes, no sampling).
- **Hidden truth and synthetic data:** pick m_f = 0.25 and K +7%, generate noisy force, compute what a plant
  would infer. Three scenarios: force only, force plus one measured diameter, force plus a known K.
- **Why a ridge:** raising friction near 0.2-0.3 lifts the whole force curve almost uniformly, like a higher K,
  so the data cannot tell them apart. The posterior is a thin curve instead of a blob.
- **What the study found beyond the planned story:** knowing K helps only partly, because the force is also
  just weakly sensitive to friction. And above about m_f 0.55 the solver's force curve has jumps from nodes
  landing on the die; I flag that region as unreliable instead of reading it as information.
- **What can go wrong:** the study is a self-consistency check: data come from the same model that is inverted
  (the "inverse crime"), so real plants will be worse. A systematic force-sensor gain error is exactly
  confounded with K and is not modelled. The prior on K and the 1% noise are assumptions; the note shows a
  sensitivity table.

## Model discrepancy and where to measure (identifiability, continued)
- **Discrepancy rows:** `identifiability.json` also feeds the inversion data that the model cannot produce: the
  bore-minimum diameter instead of the mid-plane one, and an additive force offset. The point is that the posterior
  stays narrow but moves (bias in units of the posterior std). Lead with that when quoting the 22x, which is
  precision, not accuracy.
- **K-independence is a property of the model**, not a finding: friction is defined relative to the flow stress and
  there is no hardening or temperature, so K cancels. Textbook for ring tests.
- **Where to measure** (`where_to_measure.json`, `surrogates/gp_bo/measurement_design.py`): the same grid posterior
  with a third unknown, the rate sensitivity m, marginalised over the dataset range; it compares one diameter at
  10, 20 or 30% reduction with two or three measurements. Needs a 5 x 11 solver sweep (m x m_f), cached.

## First contact with experiment
`python -m surrogates.gp_bo.validation_pop2023 --out results/validation` inverts nine published AA6068 ring tests
(Pop et al. 2023, Table 2, verbatim in `data/external/pop2023_table2.yaml`) for the Tresca factor, with brentq on
the solver as in G3. The real test is whether ONE m_f per lubricant explains the hole at all three reductions,
and whether the lubricants come out in the right order. Comparing with the authors' Coulomb mu needs the rough
conversion m_f ~ sqrt(3) mu and a different friction law read off a chart, so only agreement within a factor is
meaningful. It is a consistency check, not a validation: the flow curve and rate sensitivity of the alloy are
unknown here and it is one lab.

## Design (G5) and BoTorch
The question: which ring (OD/ID, H/ID) makes one diameter measurement most informative about friction? The
objective is the expected posterior std of m_f over all plausible true values, with the GP as forward model
(the GP's own uncertainty is added to the measurement noise, so regions the GP does not know are penalised).
It is a black box in two variables, which is what BoTorch is for: a GP on the objective plus log expected
improvement, 8 starting points and 22 steps. A dense grid search checks that BO found the optimum, and
solver sweeps of the best and standard rings check the answer without the GP. The search box equals the
training box so the GP never extrapolates. 

## Serving (G6)
`load(artifact_dir)` returns a `Surrogate` (inner-diameter change and std from the GP) and a
`FrictionEstimator`, an exact grid posterior over m_f that treats each measured diameter as a Gaussian
observation, with the GP's std added to the measurement std. `serving.check_model` exercises it.

## How to run everything
1. `python -m surrogates.gp_bo.run --data full --out results/gp`  (writes `artifacts/gp/`)
2. `python -m surrogates.gp_bo.invariance --data full --out results/gp`
3. `python -m surrogates.gp_bo.inverse --out results/gp`
4. `python -m surrogates.gp_bo.identifiability --out results/identifiability`
5. `python -m surrogates.gp_bo.design --out results/design`
5b. `python -m surrogates.gp_bo.measurement_design --out results/identifiability`  (writes `where_to_measure.json`)
6. `python -m serving.check_model surrogates.gp_bo.api:load artifacts/gp`

Solver runs are cached in `results/**/solver_cache` (keyed by case), so reruns are fast. At most 4 solver
workers are used by design. Fits take several minutes on a loaded laptop CPU.

## Things to be able to say out loud
- Why the diameter, not the force, is the friction gauge (ridge with K; flat force at high friction).
- Why the K axis is free (force scales with K, shape does not).
- Why root-finding for the inverse and BO only for the design.
- Why targets come from fixed reductions and why splits are by run.
- What the limits are: verification not validation, synthetic data, one ring, a contact artefact above m_f ~0.55.
