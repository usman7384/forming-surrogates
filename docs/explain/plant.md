# The simulated plant, explained for an ML engineer who is new to CAE

Scope: Phase 4. This is a fake forging line on a laptop. It is **simulated**: the "press" replays finite-element
runs, nothing was measured on a machine, and nothing here is validated against experiments (verification is not
validation). The project was built with AI assistance. All numbers below come from
`results/plant/drift/metrics.json` and `results/plant/nominal/metrics.json`; if a number is not in those files it is
not in this document.

## 1. The story in one paragraph

A press squeezes a metal ring once every few seconds. A load cell reports the die force and a transducer reports the
die position, at different rates and with noise. An **edge service** (a small program on a PC next to the press)
reads those signals over **OPC-UA**, works out how far the ring was squeezed, and predicts how much the ring's hole
will change (the inner-diameter change). It says **go**, **check** or **stop** before the next part. Only one part in
five is later measured with a gauge, and that result arrives several parts late. From those late measurements the
service learns the friction on the die; it can not learn friction from force alone (the project's identifiability
result, `docs/identifiability.md` once the GP work lands). Halfway through the shift the load cell starts to drift. The
drift monitor notices, and later the accuracy drops. When a prediction misses, the service saves a **repro bundle** so
R&D can re-run the solver on that part.

## 2. Vocabulary

| Term | Meaning here |
| --- | --- |
| OPC-UA | Industrial protocol. A server exposes named tags (like `Press1.Stroke.Force`); a client subscribes and is told when a tag changes. We use the `asyncua` library. |
| Tag / node id | The name of one signal, for example `ns=2;s=Press1.Stroke.Force`. All of them are listed in `docs/signal-interface.yaml`. |
| Chunk | A block of samples written into one array tag in one go. A real PLC also hands out blocks; one write is atomic, so timestamps and values can not get out of step. |
| Edge service | The program that predicts per stroke (`serving/edge/`). It talks to the model only through `serving/model_api.py`. |
| Lot | A batch of parts with the same lubricant. In this simulation every 50 parts replay one solver run, so friction and ring shape are constant inside a lot. |
| Verdict | go: predicted interval fully inside the spec band; stop: fully outside; check: anything else, including "the data looked odd". |
| Delayed label | The gauge measurement of the true inner-diameter change, sampled on 1 part in 5, arriving a few parts later (configured 3 to 8; `metrics.json` logs 2 to 7 because it counts one part fewer). |
| Acceptance metric | Share of measured parts where prediction and measurement differ by at most 1.0 percentage point, shown with a Wilson interval. |
| Wilson interval | A confidence interval for a share that behaves for small counts (with 20 measured parts a plain +/- is misleading). |
| Repro bundle | A folder with the signals, model version, config hash, prediction vs truth, a plot, and a `ringsim` case at the inferred inputs. |

## 3. What was built and why each choice

**Signal interface first (`docs/signal-interface.yaml`).** Written before any code. Every signal has a name, unit,
rate, timestamp source, node id and quality flag. The server, the simulator and the tests read this one file, and a test
fails if they drift apart. Why: on a real line the interface is the contract with the people who own the PLC.

**Press simulator (`plant/press_sim.py`).** Replays test-split solver runs. From the solver it uses the force curve and
the inner-diameter curve; everything else is invented and labelled so: plant-scale units (30 mm inner diameter,
5 mm/s die speed), 5% lot-to-lot scatter in material strength K, force noise, timing jitter, dropped samples,
bad-quality samples, a 3 ms clock offset between force and position, and a drifting force offset. Part *i* is generated
from the random stream `(seed, i)`, so a seed gives identical parts in any order. **Die temperature** is a random walk
with **no physical effect**: the solver is isothermal. It is there to prove the edge service ignores it.

**OPC-UA server (`plant/opcua_server.py`).** One variable per spec signal. The press waits for a handshake tag
(`edge_ready`) so no samples are lost at start-up; a real PLC would not wait. Security is off.

**Alignment and the reduction estimate (`serving/edge/align.py`).** The position transducer measures from the home
position, so it does not know where the part surface is. The service finds contact as the first sustained force
crossing of 5 kN, and takes the reduction as (end position - contact position) / half height. This is realistic
touch-off detection, and it is also **how the force drift leaks into predictions**: once the offset exceeds the
threshold, contact seems to start during the approach and the reduction is overestimated. The same code has a
data-quality guard (contact not where the cell expects it forces `check`) and a range guard (the surrogate was trained
for reductions up to 0.35 and clips and flags beyond that; the GP itself was trained on 0.1 to 0.3, so 0.3 to 0.35 is a
mild extrapolation).

**Prediction with unknown friction.** The model predicts at a given friction. The service keeps a friction posterior
**per lot**, starting flat, and integrates the prediction over it with a 3-point Gauss-Hermite rule, so the
prediction's uncertainty includes "we do not know the friction yet". That is why the first parts of every lot are
mostly `check`. Force is **not** fed into the friction posterior: the identifiability result says it barely
constrains friction, and a drifting load cell would bias it. A force-informed prior is designed, not built.

**Delayed labels.** The lab result carries the inner-diameter change and also the height reduction from the lab's
height gauge, so the friction update does not depend on press signals that may be drifting. (Without that, the
friction estimate quietly absorbed the reduction error during development, which is a nice example of a latent
variable hiding a sensor fault, and hid the accuracy loss.)

**Monitoring (`serving/edge/monitoring.py`).** The drift feature is the load-cell **zero reading**: mean force while
the die is still in the air. It does not depend on the ring or the lot, so a reference window (the first 100 parts)
stays valid all shift. A window of 50 parts is compared with it by PSI (5 quantile bins) and a KS test; an alert needs
both (PSI > 0.25 and KS p < 0.001). I added the "both" rule because PSI on 50 versus 100 samples is noisy by itself.
The peak force (normalised) is also monitored but is **informational**: it jumps at every lot change, so it alerts
without drift (see `peak_force_norm_alert_evaluations` in both results files).

**Repro bundle (`plant/bundle.py`, `plant/rerun_bundle.py`).** Written when a measurement arrives and the prediction
was a `go` that missed, or missed by more than its own 2-sigma interval. The `ringsim` case uses the inputs the
service *believed* (friction posterior mean, estimated reduction), because the question for R&D is "was the model
wrong or were our inputs wrong?". `python -m plant.rerun_bundle <dir>` re-runs it; a test does so on a coarse mesh.

**Dashboard (`dashboard/app.py`).** Reads the files the demo writes. A slider replays the shift. It shows the verdict,
acceptance with its Wilson band, the drift alert, and how many sampled parts are still waiting for a label.

## 4. Results (600 parts per scenario; real GP model `gp-matern52-ard`)

Sources: `results/plant/drift/metrics.json`, `results/plant/nominal/metrics.json` (commands in their `command`
fields) and `results/plant/by_split.json` (written by `python -m plant.split_report`; per-split numbers below come
from it). In the drift scenario the force offset starts creeping at part 300.

**How to read the accuracy numbers.** The GP was trained on the train split only. The demo replays the held-out test
splits: 84 runs (`test_random` 36, `test_ood_geometry` 33, `test_fine_mesh` 15), of which each 600-part shift replays 12
lots. So out-of-distribution geometries and fine-mesh runs are included on purpose, and the headline rates mix easy and
hard parts. These are simulated results; nothing is validated against a real machine.

- **Drift detection:** first alert at part 319, a delay of 19 parts after the drift started. No alert on this feature
  before the drift (`false_alarms_before_drift_primary` is 0), and none in the nominal run.
- **Acceptance** (prediction within 1.0 point of the measurement, Wilson 95% interval):
  - drift run, before the drift: 52 of 60 (87%, 76% to 93%); after it: 13 of 58 (22%, 14% to 35%); whole shift 65 of 118.
  - nominal run: 98 of 118 (83%, 75% to 89%); before the split point 52 of 60, after it 46 of 58.
  - Per split, nominal run: `test_random` 59 of 70, `test_ood_geometry` 15 of 20, `test_fine_mesh` 24 of 28 (see
    `by_split.json` for intervals). Per split, drift run before the drift: 34 of 40, 9 of 10 and 9 of 10. After the
    drift: 8 of 30, 5 of 10 and 0 of 18. The per-split after-drift numbers are confounded: which lots happen to fall
    after part 300 differs by split (18 of the 28 fine-mesh labels are after the drift), and a split has only 10 to 30
    labels, so do not read them as "fine-mesh breaks under drift". Out-of-distribution lots are never `go` (next bullet).
- **Verdicts:** drift run 241 go, 359 check, 0 stop; nominal run 392 go, 208 check, 0 stop. No `go` was ever on a part
  that was truly out of spec in the drift run (`verdict_vs_truth_in_spec`). The large `check` share in the nominal run
  is concentrated in the out-of-distribution geometry: all 100 parts of the two out-of-distribution lots are `check`
  (median predicted standard deviation about 2.3 points against a spec tolerance of 2.0, `by_split.json`), as are 28 of
  the 150 fine-mesh parts and 80 of the 350 random-split parts (friction is unknown at the start of every lot, which
  widens the interval). A `go` needs the whole mean +/- 2 sd interval inside the spec band, so a GP that is honestly
  unsure about an unfamiliar geometry can only say `check`, and its mean was also less accurate there (15 of 20
  accepted). That is the intended behaviour: the verdict follows the stated uncertainty, not the luck of the mean. In
  the drift run an additional 200 parts are `check` for a reduction outside the model's range (a force-drift symptom;
  see below). The served standard deviation is the conformally scaled one (factor 0.910, `artifacts/gp/meta.json`).
- **What the operator sees:** the alert fires well before accuracy visibly drops (see
  `results/plant/drift/figures/acceptance_trend.png`), because accuracy only suffers once the offset passes the
  contact threshold and the reduction estimate is wrong.
- **Repro bundles:** 20 written (the cap), from 53 misses; 32 of the misses were outside the model's own 2-sigma
  interval. Re-running bundle `part_00494` (`results/plant/drift/bundles/part_00494/rerun.json`; it is a drift-phase
  part with `contact_anomaly` set): at the inferred inputs the solver gives -32.0% inner-diameter change while the
  model predicted -28.9% and the gauge measured -18.9%. The solver agrees with the model (3.1 points apart) and not
  with the measurement (13.1 points apart), and the estimated reduction was 0.43 against a nominal 0.25, so the
  fault is in the inputs (the drifted force signal), not in the surrogate. That is the distinction a bundle exists to
  make. (`rerun.json` has `stopped_early: true`: the solver stopped at a reduction of 0.37, below the 0.43 requested, so the -32.0% is the value at that point; the conclusion stands.)
- **Latency:** assumed cycle time 6 s, verdict deadline 6 s. Stroke end to verdict in the drift run: p50 9.6 ms,
  p99 299 ms; nominal run: p50 8.0 ms, p99 79 ms. Inside our code (`edge_internal_ms`): p50 3.3 ms and p99 279 ms in
  the drift run (these runs shared the laptop with GNN training and a 20-seed batch, so p99 is dominated by pauses on a
  loaded machine, not by a slow code path). The GP prediction itself is p50 2.3 ms. No part missed the deadline. The definition (in the results file) includes OPC-UA delivery,
  buffer completion, alignment, features and prediction; it excludes PLC scan time and any real network, and it
  excludes the stroke's own duration because the simulator writes a stroke's chunks back to back (a burst) instead of
  in real time. The 6 s cycle is compressed 24x (0.25 s between parts). Latency varies run to run, and every latency
  number in the committed files comes from a heavily loaded laptop (GNN training and GP solver runs were running), so
  read p50 as roughly a lower-load figure and p99 as mostly scheduling noise.
- **Friction:** the final friction posterior per lot has a mean absolute error of 0.028 (`friction_mean_abs_error`,
  drift run). Lots 5, 8 and 3 end with the widest posterior standard deviations (0.18, 0.09, 0.09): with sparse, late labels and drifted reduction estimates the
  friction is not always pinned down.

**Friction estimator change (correlated surrogate error).** The friction estimator (`serving/model_api.py`,
`GridFrictionEstimator`, lead-owned) now treats the surrogate's error as one offset shared by all labels of a lot
(`correlated_model_error=True`), instead of fresh independent noise per label. The numbers above use it (code commit
7b5b7ea). Compared with the run just before the fix (code commit 27b079c, same artifact; files in git history at commit
0c66632):
- verdict counts in the seed-0 demos are unchanged (drift 241 go / 359 check, nominal 392 go / 208 check), but over
  the 20 nominal seeds of the experiment `go` fell from 6874 to 6380 and `check` rose from 5125 to 5619
  (`results/plant/experiments.json`, before and after), because the wider friction posterior keeps more intervals
  from fitting inside the spec band;
- the friction posterior is wider where the surrogate is biased, chiefly on the out-of-distribution lots (lot 3:
  standard deviation 0.013 before, 0.090 now; lot 7: 0.012 before, 0.032 now), and the mean absolute friction error
  fell from 0.052 to 0.028 (lot 9: posterior mean 0.97 before, 0.75 now, true 0.76);
- the predictive standard deviation on the out-of-distribution lots rose (median 1.7 to 2.3 points, nominal run);
- acceptance fell a little, in the seed-0 demo almost all of it on the out-of-distribution lots (nominal 102 to
  98 of 118; `test_ood_geometry` 19 to 15 of 20; drift run after the drift 17 to 13 of 58). Over 20 seeds the nominal
  acceptance went from 0.848 to 0.832 before the split point and from 0.841 to 0.808 after it;
- drift detection and false alarms are unchanged (they use force features, not the estimator).
My reading: before, the estimator "explained" the surrogate's bias on an unfamiliar geometry by a wrong friction, which
happened to make the point prediction closer to the measurement. Now the bias is attributed to the shared offset, so
the friction is closer to the truth, but the edge's prediction does not use the estimated offset (the `FrictionEstimator`
interface returns only the friction mean and sd), so it loses that accidental correction. Using the lot's offset in the
prediction is designed, not built; it needs an interface change through the lead.

**Reproducibility.** What holds: the simulator is seeded per part, labels carry a server-stamped stroke counter
(`due_index`) and the edge applies each one only after that many parts have been processed, and stroke-ends wait for
the labels the server had published before them (`labels_published`; see section 4c for the bug this fixed). So for a
fixed seed, config and GP artifact, label delays, acceptance counts, verdict counts and alert parts do not depend on
network timing. I checked this by running the full stub demo twice and comparing every non-latency block (identical),
not by repeating the GP runs. What does not hold: latency is wall-clock and varies run to run and with machine load.
The GP artifact in `artifacts/gp` is a deterministic fit with 4 torch threads (`meta.json`: `torch_threads` 4, seed 0,
torch 2.14.1), so the same artifact gives the same plant numbers, but refitting with another thread count or torch
version gives a slightly different GP and can move the counts by a few parts (an earlier artifact fitted with 8
threads moved acceptance by one or two parts). `git` in each `metrics.json` is the commit the code was run at
(7b5b7ea); the results were committed right after with identical code.

## 4b. What the 20-seed experiments show (and do not)

Source: `results/plant/experiments.json` (command in its `command` field; code commit 7b5b7ea; same GP artifact; full
data; 20 seeds x 4 scenarios x 2 recovery settings = 160 full OPC-UA demos run at press speed 0, so their latency is
meaningless and ignored). The figure is `results/plant/figures/experiments_summary.png`. Pooled counts are over 20 runs
of 600 parts. Intervals marked "bootstrap" resample whole runs (seeds), which is the honest one because the parts of a
run are not independent.

**False alarms (20 nominal runs, 451 evaluated parts each).** The zero-reading feature `f_zero_kN` and the lot-relative
force feature `peak_rel_lot` raised 0 false alarms (0 of 20 runs; Wilson upper bound 16% per run). The informational
`peak_force_norm` false-alarms 9.7 times per 100 evaluated parts, in 20 of 20 runs, because every lot change shifts
it. That is why it is not used as a drift signal.

**Detection (no recovery; delay in parts after the drift starts at part 300).**

| Drift | `f_zero_kN` | `peak_rel_lot` | `peak_force_norm` |
| --- | --- | --- | --- |
| zero offset | 20/20, median 19 (IQR 19 to 19) | 3/20, median 129 (IQR 114 to 189) | fires, but at part 309 in every scenario: the lot change at part 300, not the drift |
| gain | 0/20 | 20/20, median 39 (IQR 29 to 39) | same lot-change firing |
| friction | 0/20 | 0/20 | same lot-change firing |

Plain reading: the offset is found quickly, by the channel it was injected into. The gain error is not visible in the
zero reading; only the lot-relative force feature, which I added for this purpose, finds it, after about 39 parts.
**A true friction drift is not detected by anything in the current monitor** (the only thing that fires is the
informational feature, at the lot change). The friction posterior does see it: in the drift_friction runs the
estimated mean friction of the lots after the drift is higher by 0.22 on average, against a true rise of 0.31 (nominal
runs: -0.010 estimated, -0.005 true), but nobody monitors that quantity yet; it would be the next feature to add.

**Accuracy before and after the drift (share of measured parts within 1.0 point; pooled, bootstrap interval).**

| Scenario / recovery | before drift | after drift |
| --- | --- | --- |
| nominal | 0.832 (998/1200), 0.815 to 0.847 | 0.808 (948/1173), 0.784 to 0.830 |
| zero offset, no recovery | 0.832 | 0.330 (387/1173), 0.295 to 0.368 |
| zero offset, re-zero | 0.832 | 0.808 (948/1173), 0.784 to 0.830 |
| gain (either recovery setting) | 0.832 | 0.808 (948/1173) |
| friction (either recovery setting) | 0.832 | 0.836 (981/1173), 0.818 to 0.852 |

**Re-zeroing recovery.** After the zero-offset alert the edge subtracts the mean of the last 5 raw zero readings
(minus the healthy reference median) from the force before touch-off detection. Paired by seed, post-drift acceptance
rises by +0.478 (bootstrap 95% interval 0.436 to 0.520) compared with no recovery, to the same counts as the nominal
runs (948/1173). Re-zeroing started in 20 of 20 offset runs and in 0 of 20 nominal runs (no false re-zero). It does
nothing for gain or friction drift, and cannot: it only acts on a zero-offset alert.

**Why gain and friction drift do not hurt accuracy here.** The service uses force only for touch-off detection (a fixed
5 kN threshold crossed steeply, so a gain error barely moves it), and the surrogate does not take force as input. A
model that used force would suffer from the gain drift; this one cannot, so its immunity says nothing about a
force-based model. Friction drift is absorbed because friction is learned per lot from the delayed inner-diameter
measurements. The slightly higher friction-run rate (0.836 against 0.808) comes from a different mix of lots after the
drift, not from the drift helping.

**What produces `check` (pooled over 20 runs per scenario, no recovery).** Nominal: 5619 `check`, all
`uncertain_or_borderline`, against 6380 `go`; 19 labelled `go` parts missed by more than 1 point, and 1 `stop`. Zero
offset: 4100 `reduction_outside_model_range`, 3938 `uncertain_or_borderline`, 5 `contact_position_anomaly`, 1
`outside_data_support`. Friction drift: 5380 uncertain, 1 `outside_data_support`. The new out-of-support flag
(`in_data_support`, the censored high-friction, high-reduction corner of the training data) is raised 0 times in
nominal runs, 1 time in friction-drift runs and 362 times in offset runs (when the drifted force makes the reduction
estimate wrong); in the offset runs nearly all of those are already caught by the reduction-range check, which takes
precedence. In short: at the nominal reduction of 0.25 the corner is rarely reached, so the flag is almost never what
limits the verdicts. It is a guard for operating points this simulation hardly visits.

**What these experiments do not show.**
- **Circularity.** The press replays solver runs, and the "true" inner-diameter change, the force curve, the friction
  and the delayed lab labels all come from the same solver whose runs trained the GP (the test splits are held out by
  run, but they are the same simulator). High acceptance measures interpolation of the solver, not the behaviour of a
  press.
- **The lab label.** Besides the inner-diameter change, the label carries the part height from the lab height gauge
  (reduction +- 0.002), so the friction update does not rely on press signals. That is a design assumption; without it,
  a drifting force channel leaks into the friction estimate and hides the accuracy loss (seen during development).
- **Drift is injected in the monitored channel.** The offset drift is added to the very force channel whose zero
  reading is monitored, so 20/20 detection of it is expected, not an achievement. The gain and friction drifts were
  added to test the monitor on something it was not built for; the honest outcome is that gain needs its own feature
  and friction is not detected.
- **The collapse after the offset drift is partly by construction.** Without re-zeroing, the touch-off detector uses a
  fixed force threshold; once the offset passes it, contact appears to start in the approach and the reduction
  estimate is wrong. Re-zeroing removes that exact mechanism, so recovery is close to guaranteed here. What the numbers
  do show is the size of the effect, and that the alert comes early enough to act on.
- **The friction drift is confounded with geometry.** Lots after the drift draw from runs with higher true friction,
  and each run also has its own ring geometry (the pool has no pair of runs that differ in friction alone); some of
  those lots are out-of-distribution geometries.
- **Other limits.** One sensor model, one touch-off method, one threshold, 20 seeds of a 12-lot shift, thresholds
  (`psi_alert` 0.25, effect-size gate of 1% for `peak_rel_lot`) chosen on this simulation, nothing validated against a
  real machine. The ramp rates (0.05 kN per part, 0.2% gain per part) are my choice.

## 4c. A bug found on the way (label order at high speed)

When the harness ran the press without pauses (press speed 0), no labels were applied at all and every accuracy figure
was empty. Cause: OPC-UA delivers the queued changes of one tag after those of another within a publish, so all
stroke-end events overtook the labels the server had written before them, and the edge finished its shift before it
saw a label. At the paced demo speed (0.25 s per part) each publish held one part, which hid the problem. Fix: the
server stamps `labels_published` into every stroke-end (`docs/signal-interface.yaml`), and the edge processes a part
only after that many labels have arrived. A test
(`tests/test_plant_drifts.py::test_labels_survive_a_fast_press_over_opcua`) checks that the fast OPC-UA path gives the
same labels, delays and predictions as the offline path.

## 5. How to run it

See `docs/runbook.md`: one command for the demo, one for the dashboard, how to re-run a bundle, how to switch to the
real GP (a `--model` string, nothing else).

## 6. What can go wrong

- **Thresholds.** The drift thresholds were set for this noise level. On a real line, tune them on an in-control
  period and check the false-alarm rate on a control shift first.
- **Cold start.** Every new lot is unknown until its first measurement. The design asks the operator to measure early in
  a lot; a smarter schedule (measure the first part of each lot) is designed, not built.
- **Label starvation.** Only one part in five is measured, so the acceptance interval is wide. That is why the Wilson
  band is on the chart.
- **Out-of-distribution geometry.** The GP is unsure about unfamiliar rings, so those parts get `check` rather than
  `go`. That protects the operator but means an OOD lot gets no automation benefit.
- **Simulator realism.** Part scatter, noise levels, lots and the touch-off mechanism are my assumptions, chosen to be
  plausible, not measured. Nothing here proves performance on a real press.
- **Not built:** MQTT transport; writing verdicts back to the PLC; a force-informed friction prior; a real-time
  (non-burst) publisher.
