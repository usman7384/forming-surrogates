# Runbook: the simulated press line

Audience: someone who has this repository and a Python environment, and wants to run, watch and debug the plant demo.
Everything here is a **simulation**. The "press" replays finite-element runs from the ring-compression solver; nothing
is measured on a real machine and nothing is validated against experiments. Numbers quoted in this runbook live in
`results/plant/drift/metrics.json` and `results/plant/nominal/metrics.json`; this file does not repeat them.

## 0. What runs

```
 press simulator --> OPC-UA server (plant/opcua_server.py) --tags--> edge service (serving/edge/service.py)
   plant/press_sim.py        asyncua, port chosen by demo                 |-> model via serving.model_api
                                                                          |-> Parquet + events + repro bundles
                                                         dashboard (dashboard/app.py) reads those files
```

The tag list is `docs/signal-interface.yaml`. Both the server and the tests read it; change a tag there first.

## 1. Run the drift demo (about 3 minutes, one command)

From the repository root, with the project virtualenv active (needs `asyncua`, `streamlit`, `pyarrow`, `pandas`):

```
python -m plant.demo --scenario drift --parts 600 --model surrogates.gp_bo.api:load --artifact artifacts/gp --data full --out results/plant/drift
```

What it does: starts the OPC-UA server as a child process, connects the edge service, streams 600 parts (the force
sensor offset starts to creep at part 300), then writes everything to `--out` and prints a summary.

Control run without drift (to see the false-alarm rate):

```
python -m plant.demo --scenario nominal --parts 600 --model surrogates.gp_bo.api:load --artifact artifacts/gp --data full --out results/plant/nominal
python -m plant.split_report --root results/plant      # acceptance by replay split -> results/plant/by_split.json
```

Other scenarios (all start their drift at the same part, 300 of 600):

| `--scenario` | What drifts |
| --- | --- |
| `nominal` | nothing (control) |
| `drift` or `drift_offset` | load-cell zero offset, ramping 0.05 kN per part (`drift` is the original name and still works) |
| `drift_gain` | load-cell gain error, ramping 0.2% per part |
| `drift_friction` | true friction: lots made after the drift start replay runs with higher and higher friction (confounded with ring geometry) |

`--recovery none|rezero` selects the recovery policy (`none` is the baseline). With `rezero`, after the zero-offset alert
the edge subtracts the estimated offset from the force before touch-off detection; it does nothing for the other drifts.
Which monitor sees which drift: the zero reading sees only the offset, `peak_rel_lot` (force at a fixed die position
relative to the first parts of the same lot) sees the gain error, and nothing in the current monitor sees a friction
drift (see `docs/explain/plant.md` section 4b).

Options you will actually use: `--seed N` (different noise, same physics), `--pace-s S` (wall-clock seconds between
parts; the default 0.25 s stands for one 6 s press cycle, i.e. a 24x time compression), `--data smoke|full` (which
dataset's test splits are replayed), `--max-bundles N`, `--no-figures`, `--recovery`.

### The 20-seed experiment

```
python -m plant.experiments --seeds 20 --workers 3 --model surrogates.gp_bo.api:load --artifact artifacts/gp --data full --out results/plant/experiments.json
```

Runs 20 seeds x {nominal, drift_offset, drift_gain, drift_friction} x {recovery none, rezero} = 160 full OPC-UA demos,
at most 3 in parallel (distinct ports, press speed 0), and writes false-alarm rates, detection rate and delay per drift
type and feature, pooled acceptance with Wilson and seed-level bootstrap intervals, the re-zero recovery effect, and
the causes of `check`. About 30 s per run on the development laptop when it is busy (the committed 160 runs took 28 minutes at 3
workers; timing depends on machine load); every finished run is cached next to the output (`experiments_work/`), so an interrupted batch resumes. For a
quick try: `--seeds 1 --parts 300 --scenarios nominal,drift_offset`.

## 2. Look at the results

| File in `results/plant/<scenario>/` | What it is |
| --- | --- |
| `metrics.json` | Headline numbers in the project results format: latency p50/p99, drift detection delay, acceptance before/after with Wilson intervals, bundle count |
| `parts.parquet` | One row per part: verdict, reason, prediction, friction posterior, features, latency breakdown |
| `measurements.parquet` | One row per delayed lab measurement: residual, accepted or not, label delay in parts |
| `signals.parquet` | Aligned force and position (decimated to about 120 rows per part) |
| `drift.parquet`, `events.jsonl` | Every drift evaluation; drift alerts, labels and bundles in time order |
| `bundles/part_NNNNN/` | One repro bundle per surprising miss (see section 4) |
| `figures/*.png` | Signals with the alert marked, acceptance trend with Wilson band, latency histogram |
| `config.yaml` | The exact configuration used (its hash is stored in every bundle) |

## 3. Dashboard

```
streamlit run dashboard/app.py -- --results results/plant/drift
```

Open the printed URL (default http://localhost:8501). The slider replays the shift up to a part, so you can stop at
part 300 (nothing wrong), 330 (drift alert is up, accuracy still fine), and 450 (accuracy has dropped, verdicts are
`check`). Read it like this:

- **Latest verdict**: go (the whole predicted interval is inside the spec band), check (uncertain, or a data-quality
  guard fired), stop (the whole interval is outside the spec).
- **Acceptance**: share of measured parts whose prediction was within the tolerance of the measurement, with a Wilson
  95% interval. With one part in five measured, the interval is wide; that is the honest picture.
- **Drift alert**: the load-cell zero reading (mean force while the die is still in the air) moved away from its
  reference window (first 100 parts). It should be near zero on a healthy sensor.
- **Labels: late and sparse**: how many sampled parts are still waiting for their measurement.

## 4. Repro bundles

When a measurement shows a prediction that was a "go", or that missed by more than its own stated uncertainty, the
edge service writes `bundles/part_NNNNN/`. To re-run the solver on that part at the inputs the service believed:

```
python -m plant.rerun_bundle results/plant/drift/bundles/part_00494
```

(about 15 s at the default mesh; add `--nz 4` for a quick coarse run). It prints the solver's inner-diameter change
next to the model prediction and the measurement and writes `rerun.json`. If the solver agrees with the measurement
but not with the prediction, the model is wrong; if the solver also disagrees, the inferred inputs (usually the
reduction estimate) were wrong. `bundle.json` has the model name and version, the configuration hash, the friction
posterior, the features and the drift state at that moment; `signals.parquet` holds the raw signal window.

## 5. Run the three processes separately

Useful for debugging.

```
# terminal 1: the press (waits for the edge before the first stroke)
python -m plant.opcua_server --scenario drift --parts 600 --pace-s 0.25 --endpoint opc.tcp://127.0.0.1:4851/forming-surrogates/
# terminal 2: the edge service; it also writes metrics.json when the shift ends
python -m serving.edge.service --endpoint opc.tcp://127.0.0.1:4851/forming-surrogates/ --scenario drift --parts 600 --out results/plant/live
# terminal 3
streamlit run dashboard/app.py -- --results results/plant/live
```

`--scenario`, `--parts` and `--seed` must match on the server and the edge; the edge uses them only to compute the
end-of-shift report (it never reads the simulator's ground truth while running).

Any OPC-UA client can browse the server; the node ids are in `docs/signal-interface.yaml`
(for example `ns=2;s=Press1.Stroke.Force`). Security is off (localhost demo).

## 6. Switch the model

The edge service only knows `serving.model_api`. The model is a `module:function` string:

```
python -m plant.demo --model surrogates.gp_bo.api:load --artifact artifacts/gp --out results/plant/drift
python -m serving.check_model surrogates.gp_bo.api:load artifacts/gp      # contract test first
```

Nothing else changes. The committed results use the GP (trained on the train split only) replaying the held-out
test splits, so OOD and fine-mesh parts are included. The stub (`serving.model_api:load_stub`) is for development only:
its replayed runs are part of its own fit, so never quote its accuracy.

The friction estimator (`GridFrictionEstimator` in `serving/model_api.py`) models the surrogate's error as one offset
shared by a lot's labels (`correlated_model_error=True`, the default since commit 7b5b7ea). The committed plant
results use it; `docs/explain/plant.md` section 4 lists what changed against the earlier independent-error model.

## 8. Tests

```
python -m pytest -q tests/test_plant_spec.py tests/test_plant_sim.py tests/test_plant_opcua.py \
    tests/test_plant_edge.py tests/test_plant_monitoring.py tests/test_plant_dashboard.py \
    tests/test_plant_drifts.py tests/test_plant_split_report.py
```

## 9. When something goes wrong

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `could not reach OPC-UA server` | server child process died or port taken | run the server alone (section 5) and read its output |
| Edge prints `idle timeout` and stops early | the press stopped publishing or crashed | check the server terminal; parts processed so far are still saved only if the edge finishes normally, so rerun |
| p99 latency is far above p50 | machine busy (other jobs), OPC-UA delivery spikes | look at `transport_ms` vs `edge_ms` columns in `parts.parquet`: transport is delivery, edge is our code |
| Dashboard says "Waiting for results" | the edge has not finished the shift | wait for `metrics.json` to appear |
| `No space left on device` | the system drive is full (Python temp files go there) | free space, or set `TMP` and `TEMP` to a folder on another drive before running |
| Edge finishes a fast run (`--pace-s 0`) with zero labels | historic bug, fixed: OPC-UA delivered stroke-ends before the labels the server had written; the server now stamps `labels_published` into each stroke-end and the edge waits for that many labels | if it reappears, check that the server and edge are from the same version (an old server does not stamp the counter, and the edge then does not wait) |
| Drift alert fires in the `nominal` scenario | the thresholds are too tight for the noise | compare with `results/plant/nominal/metrics.json` (false alarms) before changing `psi_alert` |
| Many `check` verdicts at the start of every lot | the friction of a new lot is unknown until its first measurement arrives | expected; this is the identifiability result at work (force alone does not pin friction down) |

Determinism: for a fixed `--seed`, config and GP artifact (`artifacts/gp`, a deterministic 4-thread fit), the counts (labels,
acceptance, misses, alerts) should repeat exactly regardless of network timing (checked with the stub model, not by
repeating GP runs). Latency figures are wall-clock and vary with machine load; the committed latency numbers come from a
heavily loaded laptop. Refitting the GP with another thread count or torch version changes the artifact and can move the
counts by a few parts.

## 9a. Things this simulation does not do (read before demoing)

- It does not publish in real time: a stroke's chunks are written back to back, so "end of stroke" latency excludes
  the 0.8 s the stroke itself takes, and the 6 s cycle is time-compressed.
- Verdicts are not written back to a PLC, and no force-informed friction prior is used (both designed, not built).
- MQTT transport is designed, not built.
- Die temperature is a nuisance channel with no physical effect (the solver is isothermal).
