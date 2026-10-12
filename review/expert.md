# Expert review: forming-surrogates (main, HEAD a06c601)

Two independent lenses. (A) computational mechanics / metal forming. (B) ML / ML engineering.
Not a rules audit (those were separate, earlier checks). Judged: is the README's story technically sound, and what is it worth as a public demonstration of ML work on metal-forming simulation.

What I ran (short, no solver runs, ~/forming-data untouched; all via `.venv/Scripts/python`, TMP on D:):
- Re-used the committed identifiability sweep (`results/identifiability/solver_cache`, cache hit, 0.007 s) with `surrogates.gp_bo.identifiability.{Forward,posterior}` to test: force offset, ID bias, force noise level, K prior width.
- Read the cached standard-ring curves to compare `id_change_pct` with `id_min_change_pct`, and to get dID/dm_f and dID/dreduction.

---------------------------------------------------------------------------------------------------

## Scores (out of 10)

| Area | Feasibility | Correctness | Novelty / insight | Presentation |
| --- | --- | --- | --- | --- |
| Solver + verification | 8 | 6 | 4 | 8 |
| Identifiability study | 9 | 6 | 6 | 8 |
| GP surrogate + design | 9 | 7 | 5 | 8 |
| GNN | 7 | 5 | 3 | 8 |
| Plant / edge / drift | 7 | 4 | 4 | 7 |
| Engineering / repro | 8 | 7 | n/a | 7 |
| Overall project | 8 | 6 | 5 | 8 |

Honest, unusually self-critical README; the weak spots are mostly in what the numbers can bear, not in hidden dishonesty.

---------------------------------------------------------------------------------------------------

## 1. Physics (A)

### What is sound
- Norton-Hoff rigid-viscoplastic + Levy-Mises + Taylor-Hood P2/P1 + Newton with consistent tangent + updated Lagrangian midpoint is a competent, standard choice for a teaching solver (`ringsim/solver.py`). Die force by power balance rather than stress integration is a good, well-justified choice (smooth, no stress recovery noise).
- Frictionless limit is exact (+19.5% = sqrt(1/0.7)-1, `results/verify/results.json` V3 m=0), which anchors the ID axis.
- V3 trend at 30%: ID crosses zero near m_f ~0.2 (+2.1 at 0.2, -4.4 at 0.3), saturates toward ~-22 at sticking. That is the correct qualitative shape of Male-Cockcroft/Male-Depierre calibration charts (`docs/reference/calibration_chart.png`). Good.

### Where verification is weak (do not let the README imply more)
1. **V1 patch test is the weakest "pass"**: frictionless homogeneous compression. It checks incompressibility, constitutive scaling and force assembly, but exercises neither friction, nor contact, nor bulging, nor the Lagrangian update under non-uniform flow. It passes "to 0.002%" because the exact field is in the FE space; it says nothing about the hard part. Table row "force error 0.002%" reads stronger than it is. (wrong-ish framing, risky)
2. **V2 (K, V invariance) is a tautology of the formulation**: friction is defined as m_f * sigma_bar/sqrt(3), so K cancels identically. The "13 digits" result verifies code consistency, not physics. Calling it "exact to round-off" is fine; presenting "ID barely depends on strength" as a finding is not (see 2.3).
3. **"Agrees in trend with published charts" is not quantified.** No digitised Male-Depierre/Avitzur/Hawkyard-Johnson values, no error table, no stated deviation, only a chart that is the solver's own. The charts are rigid-plastic (m_rate = 0), the solver runs m_rate = 0.15, so a numeric comparison would itself need a caveat. A mechanics expert will ask "what is your m_f=0.3, 30% value versus the 6:3:2 chart?" and the repo cannot answer. (risky; cheap to fix by running m_rate small, e.g. 0.01-0.02, and tabulating against 3-4 published points.)
4. **Mesh convergence tested at m_f=0.3 (V4) and 0.7 (V4b) for ID only, nz = 6/8/12.** ID converges (0.04 pt), but the same extended file (`results/solver/verify_extended`, V4b-F) shows force 1.6599/1.6505/1.6369 for nz 6/8/12: still moving 0.8% between nz=8 and 12 and not obviously converged. The unstructured-mesh comparison (`results/solver/s4_unstructured_vs_structured.json`, key `reduction_0.3.force_diff_pct_max_abs` = 9.6%, rms 2.5%) says force at nz=8 is mesh-noise-dominated at the percent level. Yet the identifiability study treats force as a smooth 1%-noise measurement of a deterministic function. (wrong in spirit: discretisation error of force is larger than the stated measurement noise, and is a model discrepancy the "inverse crime" hides.)
5. **No convergence in time step or contact tolerance** is reported; volume drift (0.74%) is called a "spatial error", not demonstrated.
6. **No comparison against any independent code or analytic bound beyond no-slip.** V5 compares regularised Tresca with no-slip; fine as a bound check (0.43 pt gap, 0.07 with delta_rel 1e-3). Note the default `delta_rel=1e-2` leaves a 0.43-point regularisation bias at sticking, larger than the 0.3-point ID sigma used in the headline. Not a problem at m_f ~0.25, but it is a systematic error of the same order as the claimed measurement precision at high friction.

### Modelling assumptions a metal-forming expert will challenge
- **Isothermal + no hardening (n = 0 in all data).** Hot ring tests live on die chill and strain hardening/softening. With hardening the force shape depends on the flow-stress curve, which the study does not model; the K confound gets worse, not better, for force-only. So "force alone is weak" is conservative; "ID identifies m_f" survives only because ID is almost flow-stress independent, which is a known property (this is why the ring test exists). The README presents the result as the headline finding; to a mechanics reader it is a textbook rediscovery, quantified. Fine as a demo, oversold as an "insight".
- **Constant-shear (Tresca) friction factor, defined relative to flow stress.** Hot forging with lubricants/glass is often Coulomb at low pressure/Tresca-capped at high; no Coulomb-Tresca switch. Also with this definition friction scales with sigma_bar so K invariance is built in (again, 1.2).
- **Regularised Tresca at m_f -> 1 and contact artefact.** Force jumps of both signs up to ~8% above m_f 0.55 (`identifiability.json: contact_artefact_force_jump_pct_at_30` = +4.1, -6.3, +5.9, -8.0, +7.6) are a discretisation artefact of projecting nodes onto the die without remeshing/smooth contact. An 8% jump of both signs is not a subtle effect; it means force is not trustworthy at high friction (the README says so honestly).
- **Fold-over: 105/460 (23%) of runs stop early** (74/360 in train). Are these disqualifying? For ID conclusions at m_f <= ~0.5 no (runs reach 30%, ID converged). For anything claiming high-friction coverage, yes: the training table is *censored* in exactly the (m_f > 0.7, reduction = 0.3) corner (rows per reduction in train: 360/356/324 at 0.1/0.2/0.3, `results/gp/gp_accuracy.json: early_stop_handling`). The GP and the plant silently interpolate into a region with no data; the sd inflates but there is no flag. Sticking friction (m_f 0.9-1.0 in plant lots 4, 9) also is where the best-ring design folds over. The README states the facts but the design and plant narratives lean on this corner.
- **ID definition.** `id_change_pct` = mid-plane node; `id_min_change_pct` = minimum over the bore. I computed from the cached standard-ring curves at 29%: for m_f = 0, 0.3, 0.5, 0.7, 1.0 they are identical, but at m_f = 0.1 they differ by 1.41 points and at 0.25 (the headline truth) by 0.44 points. A real bore measurement is a gauge/CMM at some height; this definitional discrepancy is already larger than the 0.3-point sigma and bigger than the headline posterior implies. Effect on m_f (my run: ID shifted by +1.0 pt -> posterior mean 0.235 vs truth 0.25) is only ~0.015, but it is 3x the claimed posterior std of 0.0046. The headline 0.0046 is a precision from slope (sigma/|dID/dm_f| = 0.3/65), not an accuracy.

Verdict (1): The solver is adequate to support a teaching-scale, in-model study of ring-test *trends* at m_f <= ~0.5 with ID as the observable. It does not support force-based quantitative conclusions anywhere, nor anything at m_f > 0.55 for force, nor "agrees with published curves" beyond "same shape". Neither the contact artefact nor the fold-over disqualifies the headline (ID-based, mid-friction), but both disqualify the force-only std *curve* at m_f > 0.55 (already shaded) and the high-friction part of the design (see 3).

---------------------------------------------------------------------------------------------------

## 2. Identifiability study (A+B)

`surrogates/gp_bo/identifiability.py`, `results/identifiability/identifiability.json`, `docs/identifiability.md`.

### Is the Bayesian set-up sound?
Mechanically yes: exact 401x401 grid posterior over (m_f, log K), analytic K, PCHIP forward model with leave-one-out check (0.58% force, 0.059 pt ID), uniform prior on m_f, 100-draw expectation plus one realisation. The grid posterior is the right tool (no MCMC theatre) and the table reports both one draw and the mean. Good.

Things I checked rather than assumed (running the repo's own functions on the cached sweep, truth m_f=0.25, K=1.07):

| Check | Result |
| --- | --- |
| Reproduce force-only / force+ID | std 0.094 / 0.0046 (my noiseless draw; README 0.103 for its seeded noise draw; consistent) |
| Force noise 0.25% / 0.5% / 1% / 2% | force-only std 0.051 / 0.069 / 0.094 / 0.148, corr -0.99 -> -0.86 |
| K prior sigma 0.075 vs 0.5 | force-only std 0.094 vs 0.103 (so the K prior is not what drives "weak") |
| Additive force offset of 1% / 2% / 5% of F* (unmodelled) | force-only mean biased 0.31 / 0.32 / 0.38 (truth 0.25), std ~0.10-0.12; force+ID unchanged (0.25, 0.0046) |
| ID systematic bias of 0.3 / 0.44 / 1.0 pt | force+ID mean 0.246 / 0.243 / 0.235, std still 0.0046 |

Conclusions:
- **"Force alone constrains friction only weakly" is robust inside the model**: even at 0.25% noise the std is 0.05, so it is not a K-prior or noise-level artefact. Unmodelled offset makes force-only confidently biased, not tighter. So the qualitative claim is, if anything, conservative.
- **"One ID measurement identifies it" is correct but overconfident as stated.** sigma_mf = 0.3/65 pts/unit is arithmetic; the real uncertainty is dominated by unmodelled model discrepancy (ID definition 0.4-1.4 pt, regularisation 0.07-0.43 pt at high m_f, constitutive law, mesh), which I estimate gives m_f accuracy ~0.01-0.02, not 0.005. The honest statement is "ID gives ~10-20x tighter constraint than force, in a simulation where the solver generated the truth". The "22x" figure is a ratio of two model-conditional numbers and should not be the headline.
- **Inverse crime** is acknowledged in Limitations, correctly. Its direction matters and is not stated: it flatters ID (a tight deterministic curve) more than it flatters force (where the same solver's force noise is already larger than the assumed 1%).

### What a skeptical reviewer attacks first
1. *"The solver defines friction as a fraction of flow stress and has no hardening or temperature, so K-invariance of ID is a theorem, not a finding."* True; the docs partly say so (`ringsim.md`: "the solver proves this") but the README frames it as a result.
2. *Independent 1% noise per stroke.* Real load-cell errors are gain/offset/drift (correlated across the stroke), not white. Gain is absorbed by K (acknowledged); offset is not modelled (I tested: biases force-only by 0.02-0.13). The 10 independent 1% observations give a free statistical "shape" averaging that real data would not provide.
3. *Standard ring only, m_rate fixed and known (0.15).* The dataset varies m in 0.05-0.30; the study fixes it. Unknown m (and V^m) is a second confound on force with no mention in the identifiability doc.
4. *Truth only at m_f = 0.25.* The std-vs-true-m_f figure covers the range but above 0.55 it is unreliable by the authors' own account, and above 0.7 ID flattens.
5. *Sweep is 26 solver runs at nz=8*, i.e. discretisation error of force (~1-9%) is the same order as the 1% "noise".
6. *Interpolation LOO only for 0.1-0.5*, while the headline claims run to 1.0.
7. *Uniform prior on m_f over [0,1]*: for forging with lubricant, m_f 0.05-0.4 is typical; a uniform prior inflates "prior std 0.289" and thus the apparent information gain. (Minor.)
8. No height-gauge / reduction-measurement error in the ID observation (I checked: dID/dreduction ~ -0.2 pt per 1% at m_f 0.25, so it's a small term; but unmodelled).

Insight/novelty: moderate. The bonus finding that "K known to 2%" helps less than expected (0.105 -> 0.070) because force is itself weakly sensitive to friction (nearly uniform lift at 0.2->0.3: shape change 0.13%) is genuinely nicer than the headline, and the honest correction of the planning table is a plus.

---------------------------------------------------------------------------------------------------

## 3. GP surrogate and design (B)

### Evaluation honesty
- Fit on `train` only, targets from per-step curves at fixed 10/20/30% reductions (not from the final-QoI trap in index.csv): correct and flagged. Splits by run, three eval splits, early-stopped runs reported separately (`gp_accuracy.json: early_stop_handling`). Good.
- Baselines: cubic ridge polynomial (0.92 pt) and GBT (1.21 pt) are weak/default baselines for a smooth 5-D function; the MLP ensemble (n5: 0.10 pt) is the fair competitor and is statistically on par with the GP (0.089). So "GP is 10x better than polynomial" is the wrong headline; "GP matches a small MLP ensemble and adds calibrated UQ" is the honest one. Mean of results, not an inflated gap.
- **Coverage 99%/99%/96% for ID at nominal 95% is over-conservative, not "good".** `mean_std_id_change_pct` test_random = 0.168 vs RMSE 0.089 (sd ~1.9x error), `outputscale` 4.79 standardised vs noise 2.3e-6: a stationary kernel with one global signal variance fitted to a function whose roughness is concentrated at high friction/folding. The sd is inflated in the smooth region. Consequence: the plant's go/check logic issues too many `check` verdicts (213/600 nominal). Rows from a run share inputs (3 rows/run, 115 rows from 40 runs) so the coverage count is not 115 independent trials; CI around 99% is wider than shown. Cheap fix: report CRPS/PIT and a conformal rescale.
- Force GP: RMSE log F 1.7% vs mean sd 1.3%, coverage 87%/94%/79%: overconfident, as README says. Force rmse ~1.7% is larger than the 1% noise assumed in 2; the force GP also learns the contact jitter as noise (noise 6.4e-3).
- "ARD K length scale 83": only the invariance experiment (`results/gp/invariance_K.json`), not the main GP. The reading ("irrelevant input") is fine; the number itself is just a near-flat direction and varies by tens of percent with thread count, as stated. Not suspicious.
- OOD split extrapolates OD/ID and H/ID only; it never extrapolates m_f, m, or reduction. "OOD" is mild. (Plant verdict logic does handle reduction.)
- GP fits are thread-count dependent (documented). Fit uses Adam 300 iters x 3 restarts, no convergence diagnostics logged; for a demonstration project it is fine, for production use L-BFGS with fixed jitter and test determinism.

### Design (`design.py`, `results/design/design.json`)
- The objective is posterior std of m_f from ID(m_f) slope averaged over true m_f in [0.05,0.95]: the best ring is basically "maximise |dID/dm_f| across the range", with the GP's own sd added. BO over a 2-D box with 30 evaluations when a 21x21 grid agrees (2.35/0.78 vs 2.36/0.79) is over-engineered; the doc admits "grid search agrees". A 2-D problem does not need BoTorch; keep it as a pipeline demo, not as a result.
- Best ring (OD/ID 2.35, H/ID 0.78) lies at the OD/ID edge (box 1.6-2.4) and its solver sweep **folds over at m_f = 1.0** (`design.json: solver_check.best.complete=false`), so the check that is meant to be GP-free extrapolates its last stretch. The factor 1.66 (GP) vs 1.61 (solver) agreement therefore is partly by interpolation; and the benefit comes mostly at high friction (+19.5 to -29.8 ID range), the regime where this solver is least reliable and where ID flattens for the standard ring. Trust: "likely directionally right (wider/thicker ring has more ID travel)", not the specific optimum.
- The design ignores press-capacity (force grows with area), manufacturability, and ID measurement access. Stated in the README. Good.
- Not tested: a design of experiment over *what to measure* (ID at which reduction, multiple reductions, ID at die face vs mid-plane), which would be the real experimental-design question. Missed opportunity; same code could do it in an afternoon.

---------------------------------------------------------------------------------------------------

## 4. GNN (B)

`surrogates/gnn/{graph,model,evaluate,train}.py`, `results/gnn/*`, `docs/explain/gnn.md`.

- **Framing "works but 10x worse than the scalar model" is honest but not like-for-like.** The MLP/GP are trained on the scalar ID directly (and the MLP in n5 is evaluated at the same rows). The GNN is trained on a loss averaged over all nodes (disp, strain, force with equal weights; `train.py:compute_losses`), and the "ID" is read from a single node's displacement after training, a quantity the loss barely weights (one node of ~100-500 per graph). So ID RMSE 1.2 pt is a poor proxy for the GNN's field quality (nRMSE dr 0.061, dz 0.030 are decent). Fair comparisons: (i) field quality versus the GP on the same nodes (none exists, the GP cannot), (ii) GNN with an auxiliary ID-probe loss or a scalar ID head. Also: ID-only baselines are what the headline needs, so GNN is not on the critical path.
- **30 epochs, single seed, one run per variant**: enough to say "trains and generalises somewhat in distribution", not enough to rank variants or to say a J-loss helps (47-vs-16 comparisons are not independent, as the doc notes). The model was cosine-annealed to the shortened horizon (`--epochs 30` overrides `T_max`), so the 30 epochs are annealed; but nothing shows train/val convergence in the repo because the logs and checkpoints are git-ignored (`.gitignore`: `artifacts/gnn/**/*.pt`) and `artifacts/` holds only `gp`. The README states this.
- **Likely cause of fine-mesh failure** (hypotheses I find plausible, none tested): (a) the model sees only nz=8 graphs; fine mesh (nz=12) has ~2.25x the nodes and 0.67x edge lengths, so edge features sit outside the normaliser's range; (b) a fixed 12 hops covers a smaller physical radius on a finer mesh, and the sum aggregation changes scale with degree/edge count per physical distance; (c) the learned displacement field is a per-node map from (r0, z0, flags, s) with no mesh-size awareness. The documented hypothesis list (near-sticking friction, late stroke) fires on rule-based case attributes, not on any ablation, and cannot explain why the *fine mesh* is worst. A cheap diagnostic: evaluate on a coarser mesh (nz=6) and on a re-gridded fine-mesh field; or train with mixed nz.
- Training-mesh monoculture makes "test_fine_mesh" a test the GNN was never built to pass; framing it as "a clear failure" is fair, framing it as informative about GNNs in general is not.
- Is it worth including? As project evidence: yes, if it is positioned as "pipeline + honest negative on a limited budget", not as a result. As scientific content: weak (one seed, no physics-meaningful metric such as strain at integration points, no comparison with a field-valued baseline such as POD+GP on the same data). A POD/PCA-plus-GP (or a plain MLP on the field) baseline would give the missing like-for-like field comparison. Keep, but demote to a short section and spend effort elsewhere.
- The 8.5 ms vs 63.5 s inference/solver numbers are correctly not presented as speed-up (different work units); the "ms per stroke position" figure is batched per run on CPU.

---------------------------------------------------------------------------------------------------

## 5. Plant, edge, drift (B)

### What is real
- A real OPC-UA server/client (asyncua), chunked streams with jitter, drops and quality flags, force/position alignment, touch-off detection, delayed labels, repro bundles, PSI+KS monitor, Wilson intervals, a config-switchable model interface (`serving/model_api.py`), docker-compose written but not run (stated). This is the part closest to day-to-day ML application engineering and is well structured (`plant/press_sim.py`, `serving/edge/service.py`, `monitoring.py`).

### What is fake or circular (and the README should say it more bluntly)
1. **The "machine" replays solver runs.** True ID, force curve and friction come from the same solver the GP was trained on; the test lots are drawn from the held-out split of that solver. A good score measures interpolation of the solver, not any property of a press. (README does say "replays solver runs; there is no real machine".)
2. **Labels are solver truth plus noise.** `press_sim.py:label_for` = `id_true + N(0, 0.3)`, with reduction from the same truth +-0.002, fed to an estimator that uses the same surrogate. The friction estimator learning m_f per lot to ~0.002-0.02 for most lots (`results/plant/drift/metrics.json: friction_by_lot`) is therefore the identifiability result replayed. It is not independent validation. Even then, some lots fail: lot 5 true 0.478 vs posterior 0.679 +- 0.17; lot 9 (true 0.756) posterior ~0.97 (within the metrics file). That is the artefact/flattened-ID regime at high friction, consistent with study.
3. **The drift injected is exactly the channel monitored.** `force_offset_kN` is added to `stroke_force`; `f_zero_kN` is the mean of the same channel while the die is in air (`align.features`). Detecting a ramp in a zero-reference with PSI+KS is essentially guaranteed; the 19-part delay is mostly the 50-part window and stride 10, not a detection property. Single trial, no false-alarm rate estimate over repeated seeds, no detection power vs ramp slope, no multiplicative/gain drift, no *friction* or *material* drift (the real hazard), no drift on the ID side. `peak_force_norm` alerts without drift (45 evaluations) because lot changes shift it; the doc notes this. So the "drift detection result" is a functional demonstration, not evidence of monitoring quality.
4. **Post-drift collapse (53/60 -> 16/58) is by construction.** The offset leaks into touch-off detection (a fixed 5 kN threshold; offset up to 15 kN), which the code comments say explicitly. The service measures `f_zero_kN` and could simply re-zero but does not; the monitor alerts, nothing acts. A "drift -> automatic re-zero or fall back to `check`" loop would be the interesting piece. Note 200 parts get `check` for `reduction_outside_model_range`, which is the actual safety net; `misses_with_go_verdict = 0` is credit to that net, not to the model.
5. **Stub-vs-GP**: the interface lets both run. If the stub's predictions are an analytic or trivial function, a stub/GP comparison is circular (both defined by the same solver table). I did not find a result file comparing the two; the README does not claim one. Fine.
6. **Verdict "in spec" tolerance of 2.0 pt versus GP error 0.09 pt**: `go` = 239/239 in spec, `check` hides everything else. 100% precision of `go` at tol 2 pt is easy; the stricter question (what tolerance does the system break at) is not asked. Also the per-split table shows test_ood_geometry all `check` (median pred std 2.2 pt), which is the system working as designed but also means the whole OOD split contributes nothing to the "accuracy" claim.
7. **Label / acceptance protocol**: 118 labels, 60/58 before/after; Wilson intervals are given (good). "53 of 60 within 1.0 pt" depends on the 1.0 pt tolerance being >3 sigma of the 0.3 pt gauge noise; chosen by the author.
8. **Latency story**: p50 8.7 ms, p99 460 ms, max 1.2 s on a loaded Windows laptop; 24x time compression and burst publishing; definition excludes PLC scan time and real network. README says this; the p99/max are scheduler noise, not an SLA. Honest, but there is no per-component budget under realistic load or a soak test; do not lean on it.

### Is the drift-detection result meaningful?
As a demonstration of an end-to-end drift pipeline: yes. As evidence that PSI+KS on f_zero is a good monitor: no (single scenario, injected in the monitored feature, no ROC, no false-alarm rate over seeds). The honest sentence is "the monitor fires after ~19 parts on a 0.05 kN/part load-cell zero ramp in simulation".

---------------------------------------------------------------------------------------------------

## 6. Engineering and reproducibility

Good:
- Frozen contracts, results-JSON schema with the command and git SHA inside each file; targets from curves not index.csv; splits by run; `NumpyGP` served without torch plus a test against GPyTorch; resumable campaign; solver cache keyed by case; honest Limitations; AI disclosure in README.
- 13 test files (~860 lines). The useful ones: solver regression/patch (`test_solver.py`), numpy-vs-gpytorch GP (`test_gp_core.py`), GNN graph/model shape, PSI/Wilson (`test_plant_monitoring.py`), OPC-UA integration (`test_plant_opcua.py`). Many plant/dashboard tests are structural (does it import/parse), i.e. cosmetic relative to correctness.

Problems:
1. **The private planning notes, the agent prompts and working files, and several review files are tracked in git and contain private names that should not be public.** README and the project rules promise no company branding in public docs; if the repo goes public as-is, the planning documents are public. Remove from the public tree or `.gitignore` + history-scrub before publishing. (wrong for a public repository; cheap.) [Done before publication: the private planning notes and agent files were removed.]
2. **Missing GNN checkpoints and logs**: GNN results cannot be regenerated or audited (README says so). Commit the small `best.pt` (latent 64, <10 MB?) via release asset or LFS, plus `log.jsonl`; otherwise the nRMSE tables are unverifiable by a reader.
3. **Thread-count nondeterminism of the GP fit** (documented): plant pass counts "can move by a part or two". Pin threads in `fit_gp(threads=8)` default at least in CI and record thread count and torch version in `meta.json`; for a demonstration project, deterministic seeding of the optimisation matters more than 5 percent RMSE.
4. `results/` has 306 tracked files, mostly solver caches; fine for provenance but the repo is large and noisy for a reviewer; put caches under a single `results/**/solver_cache/` listed in README.
5. The `fit_gp`, `run_bo` and others hard-code `torch.set_num_threads(8)` (design.py:85); on a 2-core CI machine this oversubscribes.
6. Docker stack unrun; README says so. OK.

---------------------------------------------------------------------------------------------------

## 7. Project value

### Three strongest elements
1. **Epistemic honesty throughout** (verification vs validation, "designed, not built", AI disclosure, contact-artefact shading, force vs ID mesh sensitivity, honest negative GNN). Rare, and exactly what simulation-software readers look for in ML work.
2. **A coherent end-to-end chain**: own FE solver -> dataset with fold-over flagged -> GP with UQ and a model-API contract -> edge service on OPC-UA with delayed labels, drift, repro bundles. The `ringsim`-case repro bundle ("the faulty input, not the surrogate") is a good applied idea.
3. **The identifiability analysis as a worked Bayesian-inference example with its own corrections** (K-known helps less than expected; step table recomputed against the planning claim; sensitivity to analyst choices). Clean, small, exactly reproducible.

### Three weakest / most attackable
1. **Headline over-reads an in-model fact.** "22x" and "0.0046" rest on a solver-defined truth with no hardening, no temperature, mid-plane ID definition, and white 1% noise; ID-vs-force asymmetry is textbook knowledge. Present it as a method demo ("how to quantify it"), with ID accuracy ~0.01-0.02 once definitional discrepancy is added.
2. **Verification is thin where it matters**: patch test without friction, no quantified comparison against published ring-test curves, force not mesh-converged (0.8-9.6%), 23% fold-over with censored high-friction data, contact jumps up to 8%. A forming-simulation specialist will find this in ten minutes.
3. **Plant results are circular and the drift demo is a self-fulfilling prophecy** (replay of the training solver; drift injected into the monitored channel; failure after drift caused by an uncorrected touch-off; one scenario). Interesting plumbing, weak evidence.

### Questions a technical reader would ask that the owner could not answer from the docs
- "What is the ring-test m_f from the standard chart at 30% for your 6:3:2 and how far off are you?" (not quantified.)
- "How would hardening or die chill change both claims?" (n=0 everywhere; no experiment.)
- "Is mid-plane ID what a CMM measures; what is the error from picking min-ID?" (0.4-1.4 pt at m_f 0.1-0.25; not discussed in the headline.)
- "Why Matern 5/2 ARD GP rather than an MLP ensemble, which scored the same?" (answered only implicitly: UQ.)
- "Why is your GP 2x too wide at 95%?" (not explained.)
- "What does your drift monitor do on a friction drift, a gain drift, or a seed other than 0?" (not run.)
- "How would this run with a real OPC-UA PLC, security policies, store-and-forward?" (designed, not built.)
- "Why does the GNN fail on a finer mesh; what would you change?" (hypotheses only.)
- "How would you retrain/version the GP when new labelled parts arrive; where is the model registry?" (bundles record version, but no registry/CI flow.)

---------------------------------------------------------------------------------------------------

## Top 10 improvements (ranked by value / effort)

| # | Change | Effort | Rerun? |
| --- | --- | --- | --- |
| 1 | Remove or scrub the private planning notes, agent files and private names in the process records and `review/` from the public tree (gitignore + history clean). | trivial | no |
| 2 | Reword the headline: say "force constrains m_f to about +-0.1; one ID gauge to about +-0.005 in-simulation, +-0.02 once an unmodelled ID bias of 1 point is included"; add a "model discrepancy" row (ID definition, offset) to `identifiability.json` using the script I used (30 lines). Add "why the ring test uses ID" textbook note and the hardening caveat. | cheap | trivial rerun |
| 3 | Quantify V3 against 4-6 digitised published points (Male-Depierre/Avitzur) with m_rate small (0.01-0.02), report errors in a table; downgrade V1 wording to "homogeneous-field test". | cheap-medium | small solver rerun |
| 4 | Add conformal rescaling / PIT / CRPS to the GP report and discuss the 2x conservatism; report an MLP-ensemble comparison in the GP table instead of poly/GBT. | cheap | rerun `gp.run` (minutes) |
| 5 | Plant: run drift detection over, say, 20 seeds and 3 drift types (zero offset, gain, friction/lot-level shift); report false-alarm rate and detection delay distribution; add re-zeroing using `f_zero` and show post-drift accuracy recovery. | medium | rerun plant (about 2.5 min each) |
| 6 | Flag the censored region (m_f > 0.7, reduction 0.3) in the GP/plant API (extra "outside data support" reason) and rerun the design with the solver check on a common m_f range, or restrict objective to m_f <= 0.8. | medium | rerun design check (cache helps) |
| 7 | GNN: add a like-for-like field baseline (POD+GP or MLP on field) and an ID auxiliary loss; test coarse-mesh/mixed-nz training to confirm the fine-mesh hypothesis; commit `best.pt` + `log.jsonl`. Otherwise shrink the GNN section to one paragraph. | medium-high | multi-hour retrain |
| 8 | Fix GP determinism: fix threads/seed and record versions in `meta.json`; add a test that two fits agree within tolerance. | cheap | refit GP |
| 9 | Extend the identifiability study to the two real design questions: measure ID at which reduction(s)/location, and unknown m_rate/hardening as a third latent. | medium | uses existing cached sweeps plus a few new runs |
| 10 | Add a short "what this does not show" box at the top of the README: truth = same solver; no hardening/temperature; ID at mid-plane; plant is circular; force artefact above m_f 0.55. | trivial | no |

---------------------------------------------------------------------------------------------------

## Overall verdict

A well-organised, unusually honest project with a real engineering backbone (solver, dataset, GP, OPC-UA edge, bundles). The solid claims: the solver reproduces the right qualitative ring-test behaviour; force is a weak friction gauge and ID a strong one *inside this model*; the GP matches a small MLP and has usable (conservative for ID, overconfident for force) uncertainty; the plumbing works. The overreaches: headline "22x" as an insight (it is a textbook property of the test, quantified in a self-consistent simulation), the "verified" label resting on an easy patch test and an unquantified chart comparison, the GNN comparison framed on a metric the GNN is not trained for, and the plant results being circular. None of this is dishonest, because nearly every weakness is already disclosed in Limitations; it is that the headline numbers are given more prominence than the limitations. For ML architecture and ML application engineering it is a strong showing (breadth, rigour of presentation, MLOps shape); for a CAE-methods reader the verification depth will be the first thing questioned. Fix the public-tree leak (#1) and reframe the headline (#2, #10) before sharing; with #3-#5 done it becomes a strong piece.

Overall: 8/10 feasibility and presentation, 6/10 correctness of the claims as worded, 5/10 novelty.
