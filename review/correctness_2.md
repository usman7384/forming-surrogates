# Correctness review 2 (changes since review/correctness.md)

Reviewer: independent agent, fresh context, read-only on the repository (this file is the only one written).
State reviewed: `main` at f65ef55. Interpreter `.venv`, `FS_DATA_ROOT` set to the read-only full dataset.
Scratch scripts were run outside the repository and deleted afterwards. Their key output is quoted below.

## Verdict: **correct with caveats**

No bug changes a headline number. The checks below were recomputed independently and all agree:

- The new correlated-error friction estimator is the exact joint-Gaussian posterior.
- The Pop 2023 inversion reproduces with my own solver calls to the 4th decimal.
- The where-to-measure value after the V/h0 fix reproduces (0.0211 against 0.0212).
- The design breakdown recomputes from its own per-truth lists.
- The hero figure's assert holds bit for bit.
- The plant counts add up.
- Tests pass with both the full and an empty data root.

The caveats:

- One README number contradicts its results file (GNN inference 8.5 ms; the file says 18.5 ms).
- Two README statements are worded more strongly than their files support: the censoring share, and why friction error fell in the plant.
- Four findings of the first review are still open, all without effect on numbers.

## Tests

| run | result |
| --- | --- |
| `python -m pytest -q`, `FS_DATA_ROOT` = full dataset | **91 passed**, 4 warnings, 99.7 s |
| same, `FS_DATA_ROOT` = an empty folder | **91 passed**, 4 warnings, 94.1 s (nothing skipped, nothing written to the folder or the tree) |

---

## Findings ranked by severity

Severity scale: **R** changes a reported number · **B** bug with no effect on reported numbers · **Q** questionable
modelling choice · **D** doc mismatch · **N** nit.

| # | sev | where | finding | effect on README |
| --- | --- | --- | --- | --- |
| 1 | D (README number contradicts its file) | README.md:162, docs/explain/gnn.md:114 | GNN inference "about 8.5 ms per stroke position". `results/gnn/eval_base_full/metrics.json` `metrics.inference.ms_per_sample_batched_run` = **18.53** (jloss 15.44, idloss 11.16, mixmesh 11.66). 8.49 exists only in the file at eecd772; 0e6b79c ("metrics unchanged") regenerated it. | Replace with about 11 to 19 ms (or quote the base value, 18.5). The 63.5 s solver time is correct. |
| 2 | D | README.md:189-194 (finding 4 paragraph) | The friction MAE fell from 0.052 to 0.028, but the out-of-range lots are not what drove it. Per-lot error (posterior mean minus truth), old (0c66632) → new: lot 9 (test_random, m_f 0.756) **+0.213 (z = +3.4) → −0.006**; lot 8 +0.089 → +0.049. On the out-of-range lots, lot 7 went −0.017 → −0.004, but **lot 3's error grew in size and changed sign, −0.023 → +0.027**. The "bias absorbed into friction" explanation of the 19/20 → 15/20 drop does fit the residuals: on both out-of-range lots the old m_f sat low (z −1.9 and −1.4), and the old residuals were less negative (lot 3 mean pred−meas −0.20 old against −0.50 new; lot 7 −0.67 against −1.07). docs/explain/plant.md:151 names lot 9; the README does not. | Numbers are correct. Wording: the overconfidence the fix removed was worst on an in-distribution high-friction lot, not only on out-of-range geometry. |
| 3 | D | README.md:21 | "about a quarter of the high-friction runs stop early". `results/solver/censoring.json`: fraction reaching 0.35 is 0.535 (m_f 0.6–0.8, 99 runs) and 0.533 (0.8–1.0, 92 runs), so about **47%** of runs with m_f > 0.6 stop early. All 105 early stops are at m_f > 0.4. A quarter (105/460) is the share of *all* runs, which lines 223 and 229 state correctly. | Line 21: "about a quarter of all runs, nearly half of those with m_f > 0.6". |
| 4 | D | README.md:14 | "one friction factor per lubricant reproduces the measured inner diameters ... to within 0.02 to 0.03". 0.017–0.030 is the **spread of m_f**, not an inner-diameter error. At each lubricant's mean m_f the solver misses the measured ID change by −2.07 to +1.13 points (mineral oil 40%: −2.07 points = −0.31 mm; dry: −0.55 / +0.23 / +0.70). Line 97 states it correctly. | Reword line 14 ("... with friction factors that agree within 0.02 to 0.03"). |
| 5 | D (old finding 8, open) | serving/edge/service.py:42 and :159, docs/explain/plant.md:58 | RED_MAX = 0.35 with the comments "reduction range the surrogate was trained on" and "trained on reductions up to 0.35". The GP is trained on 0.1–0.3. | None (see correctness.md: affected parts are already `check`). |
| 6 | B (old finding 1, open) | ringsim/solver.py:9 | The module docstring still claims "the consistent tangent ... of the friction law". The comment at :262 is honest ("flow stress frozen inside the friction tangent"). | None. |
| 7 | N (old finding 2, open) | ringsim/verify.py:118 | V2 still prints round-off with `.1e`. results/verify/report.md has "1.4e-13 / 2.9e-13" against docs/reference/verify_report.md "6.8e-14 / 0.0e+00", so `ringsim.verify` still does not reproduce the reference report byte-exactly (rule 3's wording). | None. |
| 8 | Q (old finding 4, open) | surrogates/gp_bo/run.py:290-294 | The choice between conformal and raw served sd is still made on test_random/test_ood coverage. Recomputed: coverage95 is identical for raw and scaled sd on every split (0.983 / 0.991 / 0.943), so the choice changes nothing. | None. |
| 9 | N (old finding 9b, open) | surrogates/gnn/evaluate.py:143 | The predicted-mesh J uses the predicted step-0 mesh `P[0]` as reference, unlike the training loss (p0). | Affects only `mean_abs_J_minus_1_pred` if the step-0 prediction is nonzero. |
| 10 | N | serving/model_api.py:95 | `sigma_pct = 0` raises ZeroDivisionError in the default (correlated) mode; the old model accepted it when sd > 0. The plant always sends 0.3. | None. |
| 11 | N | serving/model_api.py:100 | `A − B²/(1+S)` cancels when model sd ≫ gauge sigma. Stress test with a constant shared bias (exact answer known): max posterior difference 8e-10 at sd/sigma = 5e3 (n = 50), 2e-6 at 5e4 (n = 500), 9e-4 at 5e6 (n = 2000; std 0.2908 against 0.2915). The plant ratio is about 0.5–8, so this is irrelevant here. | None. |
| 12 | N | tests/test_friction_estimator.py:15-19 | The fake surrogate's sd depends on m_f only, so the brute-force test never has sd changing between the labels of one update set (the general s_i case of Sherman-Morrison). I checked that case separately (below); adding a reduction-dependent sd to the fake would close the gap. | None. |
| 13 | N | surrogates/gp_bo/api.py:6-8 | The docstring still says the estimator "adds the predictive variance to the measurement variance", which describes the old independent model. | None. |
| 14 | N | serving/edge/service.py:213,224 | `label_delay_parts` is reported one lower than configured: delay = (len(records) − 1) − index, so config (3, 8) is logged as 2–7 (metrics.json `delay_parts_min/max` 2/7). docs/explain/plant.md:31 says "2 to 8". | None (the delay is reporting only). |
| 15 | D | README.md:153 | "cut the fine-mesh field error by 60 to 70%": jloss → mixmesh test_fine_mesh nRMSE falls 64% (dr), **72%** (dz), 66% (strain). | "60 to 75%". |
| 16 | D | README.md:157-158 | The in-distribution factors (3x / 1.6–1.7x) are base-GNN against POD, but "0.131 against 0.214" is the **jloss** model's OOD dr (base: 0.135). The variant switches silently. | Name the variant. |
| 17 | D | README.md:238-239 | "a fresh clone ... regenerated ... every downstream metric exactly": `results/reproduction/clean_clone_check.json` (committed e051d4a) predates the regeneration of plant metrics/by_split (b1dc6ba), design.json (00ca28e) and eval_base_full (0e6b79c). | Say the check covers the state at e051d4a. |
| 18 | N | results/plant/drift/bundles/part_00539/ | A leftover from 51c76e9: 21 bundle folders exist, while `bundles.written` = 20 and events.jsonl references 20. | None. |
| 19 | N | docs/identifiability.md:66 | "the planning table in the explainer" refers to a document removed in the cleanup (the private planning notes). | None. |
| 20 | N | README.md:236 | "an earlier 8-thread fit differed by a few percent ..." is in no results file (only docs/explain/gp_identifiability.md:38). | None (provenance only). |
| 21 | Q (plausibility of data) | data/external/pop2023_table2.yaml rows 2 and 5 | Volume implied by the tabulated diameters and heights, (d1e² − d1i²) h1 / (30² − 15²) 10 − 1: dry 0.2/0.3/0.4 = +0.1 / **+3.9** / +1.1%; mineral oil −1.2 / **+5.4** / +1.9%; graphite −3.6 / −0.9 / −0.5%. The two 30% rows (dry, mineral oil) are the outliers. Barrelling makes the measured outer diameter overstate volume, so this may be real, but it is also where predicted-minus-measured OD is negative (−0.74, −1.74 points; the other 7 rows are positive). Worth one check against the paper. Everything else is internally consistent: ID decreases with reduction for all three lubricants; h1 agrees with eps exactly; the m_f ordering dry > oil > graphite matches the authors' mu order. The authors' mu is not monotone in reduction (dry 0.15/0.20/0.17), and mineral oil and graphite share mu = 0.09 at 30/40%, which is consistent with chart reading. | None on numbers; a plausibility flag. |

---

## 1. Friction estimator (serving/model_api.py:59-110)

**Math: correct.** For C = D + s sᵀ (D = diag σᵢ²), Sherman-Morrison gives rᵀC⁻¹r = A − B²/(1+S) and
log|C| = Σ log σᵢ² + log(1+S), with A, B, S as in the docstring. Σ log σᵢ² is the same at every grid point and cancels.
The posterior is recomputed from the prior and the running sums at every update, so nothing accumulates.

Brute-force check against `scipy.stats.multivariate_normal` on the full covariance, per grid point. The fake surrogate
has a non-linear mean and an sd that depends on m_f **and on the label's reduction**; σᵢ is drawn per label:

| case | max \|p_est − p_bruteforce\| |
| --- | --- |
| sd varies per label, 6 labels | 2.0e-15 |
| sd = 0 | 1.9e-19 |
| sd × 30 (large model error) | 4.9e-13 |
| 300 labels | 1.2e-50 |
| σ = 1e-3, 20 labels | 0 |

- `correlated_model_error=False` is line for line the pre-fix code (`git show c72f52c:serving/model_api.py`), and
  matches an independent-normal sum to round-off.
- The default is `True`, as the docstring says. `api.load` constructs it with n = 201 and the default.
- **Tests:** `test_correlated_update_equals_joint_gaussian` is a genuine brute-force MVN comparison, with σ varying
  per label. Its gap is finding 12. `test_single_label_matches_independent_model` is right: with one label the two
  models coincide. `test_repeated_labels_do_not_shrink_below_the_model_error` checks the qualitative claim with a
  relative bound (> 2.5x), which is adequate.
- Modelling note (Q, no action needed): "fully correlated" uses the GP's own sd at each label's input as the
  per-label loading. That is the conservative limit for labels with nearly identical inputs, which matches a lot
  (fixed ring and m, reductions within ±0.01).

## 2. Pop et al. 2023 inversion (validation_pop2023.py, pop2023_table2.yaml)

**Code: correct.**

- ID% = 100 (d1i/15 − 1); the resolution interval is ±0.05 mm on d1i.
- The interval is mapped correctly: a larger hole means lower m_f, so `id_change_pct_hi` gives the lower bound.
- The figure's error bar uses 100·0.05/15.
- The case is the standard ring: od_id 2, h_id 2/3, matching 30:15:10 mm.
- brentq brackets [0, hi], where hi is the largest sweep node that reaches the row's reduction; xtol is 1e-3 on the
  solver.
- The sweep-PCHIP inversion agrees with brentq to ≤ 2e-4 in every row.
- The script's own consistency check `id_change_pct_solver_at_mf_check` is within 0.03 points of the target in every
  row.

**40% reached?** In the cached sweep, m_f ≤ 0.425 reaches 0.40 and 0.45 stops at 0.39. My direct runs agree: m_f
0.425 reaches 0.40 and 0.45 stops early at 0.39. All inferred m_f (0.24–0.37) lie inside the bracket.

**Independent recomputation** (my own `RingSolver` calls, default numerics, m = 0.05, project cache not used; secant
between two runs):

| row | target ID% | runs (m_f → ID%) | my m_f | file brentq / sweep | my resolution interval | file interval |
| --- | --- | --- | --- | --- | --- | --- |
| dry, 30% | −12.000 | 0.35 → −11.067, 0.375 → −13.080 | 0.3616 | 0.3615 / 0.3614 | [0.3574, 0.3657] | [0.3573, 0.3655] |
| mineral oil, 40% | −7.333 | 0.25 → −7.095, 0.26 → −8.307 | 0.2520 | 0.2520 / 0.2519 | [0.2492, 0.2547] | [0.2492, 0.2547] |

**sqrt(3)·mu comparison:** the ratios are 1.194 / 1.864 / 1.834, matching README "1.2" and "about 1.85". The
caveats are stated in the file notes.

**README §3 table matches the file** to every printed digit:

- per-row m_f, means 0.359 / 0.269 / 0.249, spreads 0.018 / 0.030 / 0.017;
- resolution half-widths 0.0027–0.0066 ("0.003 to 0.007");
- m = 0.02 shift ≤ 0.026;
- OD residuals −1.74 to +1.98 ("−1.7 to +2.0"). "No consistent sign" is accurate but generous: 7 of 9 are positive.

Wording issue: finding 4. Data plausibility: finding 21.

## 3. Where to measure after the V/h0 = 1.5 fix

`measurement_design.py:44,67` adds m·log 1.5 to log F*. Truth and model use the same forward model, and the docstring
is now correct. My independent posterior:

- same cached sweep, read directly;
- PCHIP in m_f then in m, on finer grids: m_f 1001 points, m 51 points;
- log K marginalised analytically.

| truth (m_f, m = 0.15) | quantity | file | independent |
| --- | --- | --- | --- |
| 0.5 | force + ID@30%, m unknown | 0.02116 | **0.02113** |
| 0.5 | same, m known | 0.00791 | 0.00791 |
| 0.25 | force + ID@30%, m unknown / known | 0.00625 / 0.00462 | 0.00625 / 0.00462 |
| 0.1 | force + ID@30%, m unknown / known | 0.00345 / 0.00343 | 0.00345 / 0.00344 |
| 0.5 / 0.25 | force only, m unknown | 0.1238 / 0.1403 | 0.1208 / 0.1397 (grid-resolution difference, ≤ 2.4%) |

The 0.0212 in README:75 and docs/identifiability.md:148 is the value the first review predicted for the fix.

## 4. Design breakdown (results/design/design.json)

Recomputed from the stored per-truth lists. The mean of `solver_check.standard.std_by_true_mf` is 0.015729 and of
`best` is 0.009264, which equal the stored objectives; the factor is 1.698.

| quantity | value | README |
| --- | --- | --- |
| solver factor, truths ≤ 0.6 | 1.239 | 1.24 |
| solver factor, truths > 0.6 | 1.919 | not quoted |
| share of the standard-ring average from truths > 0.6 | 0.762 | not quoted |
| GP per-truth means (standard / best) | 0.016912 / 0.010342 (= `objective`; factor 1.635) | 0.0169 / 0.0103, 1.64 |
| GP factor, truths ≤ 0.6 | 1.201 | 1.20 |

Both sweeps are complete to m_f = 1 (common range [0, 1], no extrapolation). Old finding 6 is fixed: README:133-135
now says the gain is mostly from high friction.

## 5. Hero figure

`hero_figure.py:34-38` replays `default_rng(0)` in the same order as `identifiability.main`. `expected_std` uses its
own generator, so it does not consume the main stream. I ran the script against a scratch copy of
identifiability.json and the solver cache. The assert passed, and the values are identical to the file:

- force_only 0.1033787938475441
- force_plus_id 0.0045939694339860865

Nit: the seed is hard-coded, not read from the results file. A rerun of identifiability with another `--seed` would
make the assert fail loudly, not silently.

## 6. Plant results after the estimator fix

**Counts add up.**

- **Drift run:**
  - verdicts: go 241 + check 359 = 600;
  - verdict reasons: 241 + 200 + 159 = 600;
  - acceptance: 52/60 + 13/58 = 65/118; misses 53 = 118 − 65;
  - after-alert 12/55 + start-to-alert 1/3 = 13/58.
- **Nominal run:** go 392 + check 208 = 600; acceptance 52/60 + 46/58 = 98/118; misses 20.
- **by_split.json:** per split the counts sum to the run totals in both scenarios (59+15+24 = 98, 42+14+9 = 65,
  34+9+9 = 52, 8+5+0 = 13; verdicts 270+122 = 392, 80+100+28 = 208, 198+43 = 241, 152+100+107 = 359).
- **experiments.json:**
  - 160 runs = 4 scenarios × 2 recovery settings × 20 seeds;
  - pooled re-zero gain 948/1173 − 387/1173 = 0.478, equal to the paired mean +0.478, bootstrap [0.436, 0.520];
  - false-alarm and detection counts match README:181-188 and docs/explain/plant.md:183-216.

**Friction:** `friction_mean_abs_error` recomputes to 0.02795 from the 12 lots, and every lot's |z| is ≤ 1.13. The
old value at 0c66632 is 0.0515.

**Intervals:** Wilson recomputed with scipy (52/60 → [0.7583, 0.9309]; 13/58 → [0.1359, 0.3466];
998/1200 → [0.8094, 0.8518]; 387/1173 → [0.3036, 0.3573]; 15/20 → [0.5313, 0.8881]); all match. The cluster
bootstrap is sometimes *narrower* than Wilson (before drift [0.815, 0.847] against [0.809, 0.852]). That is plausible:
each seed replays the same 84-run test pool, so between-run variance is below binomial. But it means the bootstrap
measures seed-to-seed variability on a fixed pool, not population uncertainty, and with 20 clusters percentile
intervals run slightly narrow (Q, no action).

**Observation (no error):** gain-drift runs give exactly the nominal acceptance and verdict counts (948/1173; 6380 go,
5619 check, 1 stop). Predictions use position-based features only, and the 0.2%/part gain does not move the 5 kN
touch-off, so this is expected. Re-zero after an offset alert also returns exactly to the nominal counts, which is
the "by construction" the README admits.

**Docs agree with files:** README §6 and docs/explain/plant.md quote the same numbers as metrics.json /
by_split.json / experiments.json (52/60, 13/58, 98/118, 208, 319/19, 9.6/299 ms, 0.028, 0.052, 19 → 15 of 20,
6874 → 6380 go).

**The bias-absorption claim:** see finding 2. It is consistent for the acceptance drop, but the MAE improvement comes
mostly from lot 9 (in-distribution), not from the out-of-range lots.

## 7. Status of the findings in review/correctness.md

| # | finding | status |
| --- | --- | --- |
| 1 | solver.py:9 docstring claims a consistent friction tangent | **open** (finding 6) |
| 2 | verify V2 round-off digits not reproducible | **open** (finding 7) |
| 3 | where_to_measure V/h0 = 1 | **fixed** (code, docstring, file 0.0212, README, docs) |
| 4 | served-sd rule decided on test splits | **open** (finding 8; no number changes) |
| 5 | force non-monotone from m_f ≈ 0.43 | **fixed/documented** (README:60-62, shading from 0.45) |
| 6 | design objective dominated by m_f > 0.6 | **fixed/documented** (README:133-135, design.json breakdown) |
| 7 | GridFrictionEstimator independent errors | **fixed** (§1; documentation of the effect: finding 2) |
| 8 | RED_MAX 0.35 vs GP trained to 0.30 | **open** (finding 5) |
| 9a | README "60 extra runs" = 51 + 9 | **documented** (README:153 "some of them held out for validation") |
| 9b | evaluate.py J reference | **open** (finding 9) |
| — | missing tests recommended in §8 of the first review | #5 (estimator vs closed form) **added**; #1–#4 (FD Jacobian, force = reaction, golden solver run, analytic-K posterior) **not added** |

## 8. Repository cleanup

- **No tracked code** (py, yaml, toml, cfg, ini, sh, ps1, requirements) imports or reads a removed or moved path:
  the agent instruction files, the private planning notes, status/, Makefile, .github, docker-compose, prompts.
  - Three docstrings name the results-JSON schema document, which existed at the time (later removed; the schema is now described in the docstrings).
  - `.gitignore:10 .claude/` is harmless.
- **Links:** all 63 relative links in README.md and every tracked .md resolve to tracked files. The README anchors
  `#how-this-was-built` and `#limitations` match headings.
- **No results-JSON `command`** refers to make or a removed path.
- **No branding hits** in tracked files (`git grep -i` for the two names).
- **No reader-facing doc tells the reader to run `make`.** The remaining `make` and docker-compose mentions were in
  the process records (plan, contracts, task board), which were later removed from the published repository.
- Stale reference: finding 19.

## 9. README number trace

About 170 numbers were checked against results files. All match after rounding except findings 1, 3, 15 and 16, plus
17 and 20 (provenance only). Finding 4 is a mislabelled quantity, not a wrong value.

## Reproductions (all short)

- **Estimator:** brute-force MVN per grid point, five scenarios (§1), plus a cancellation stress test with a known
  closed form.
- **Pop rows:** 6 direct `RingSolver` runs, about 1 min on 4 workers: m = 0.05, m_f 0.35 / 0.375 to 30%, 0.25 / 0.26,
  0.425, 0.45 to 40%.
- **Where to measure:** independent grid posterior from the cached sweep.
- **Design:** recomputed from the stored lists.
- **Hero:** script run against a scratch copy.
- **Plant:** old results via `git show 0c66632:results/plant/...` compared with HEAD parquet and JSON.
