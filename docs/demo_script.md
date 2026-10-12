# Demo scripts (2-3 minutes each)

Two variants of the same project, one per audience. Everything shown comes from files in `results/`. Say it once,
early, in both: **this is a simulation project built with AI assistance; the solver is verified against exact
solutions, and has only one point of contact with experiment (see the validation shot).**

Before recording: activate the venv, open the repo in an editor, and start the dashboard once so it is warm:
```bash
streamlit run dashboard/app.py -- --results results/plant/drift
```

## Variant A: for a deep-learning surrogate team
Lead with the ML evaluation, because that is what they judge.

| Time | Show | Say |
| --- | --- | --- |
| 0:00-0:20 | `README.md` top | A ring is squeezed between two dies; the inner diameter tells you the friction. I generated 460 finite-element runs and trained surrogates on them. The interesting part is how they compare against honest baselines. |
| 0:20-0:50 | `results/gp/figures/parity.png`, the GP table in the README | A Gaussian process predicts inner diameter to about 0.09 points in distribution. A plain MLP ensemble does about as well (0.10), so the GP earns its place through its uncertainty and its use in Bayesian design, not through accuracy. |
| 0:50-1:40 | `results/gnn/figures/field_comparison.png`, `results/gnn/variants_comparison_full.json` | The mesh GNN predicts whole fields. A POD-plus-GP field model with no message passing beats it 1.6 to 3 times in distribution, so for this simple geometry the GNN's value is not shown; it only wins on out-of-range geometry. It failed on a finer mesh; training on 60 extra fine-mesh runs cut that field error by 60 to 70%, so most of the failure was a mesh-density shift, with no test leakage (checked). |
| 1:40-2:20 | `results/identifiability/figures/hero_posterior.png` | The physics question: force alone barely pins down friction (posterior std 0.10); one inner-diameter measurement brings it to 0.0046. That is precision, not accuracy: a 1-point bias in how the diameter is measured moves the answer by several standard deviations. |
| 2:20-2:50 | `README.md` "What this does not show" and "How this was built" | What I would do next with a real geometry: a cross-section that a fixed-grid POD cannot represent, where a GNN should earn its keep; several seeds; GNN ensembles for uncertainty. |

## Variant B: for a forming-simulation team
Lead with the solver's limits, then the identifiability, then the plant loop back to R&D.

| Time | Show | Say |
| --- | --- | --- |
| 0:00-0:30 | `results/verify/calibration_chart.png`, `docs/reference/verify_report.md` | A small rigid-viscoplastic axisymmetric solver with regularised Tresca friction. Verified: exact homogeneous solution to 0.002% in force, mesh convergence of the inner diameter to 0.02 points. Its weak spots, stated up front: force is less mesh-converged than the diameter, there is no remeshing so about a quarter of the high-friction runs stop early, contact steps of up to 8% in force at high friction, isothermal, no hardening. |
| 0:30-1:20 | `results/identifiability/figures/hero_posterior.png`, then the model-discrepancy table in `docs/identifiability.md` | Can a plant read friction from the press force? Force and material strength are confounded (correlation -0.95): std 0.10. One inner-diameter measurement: 0.0046. But a bore-minimum instead of a mid-plane diameter biases the answer by 1.4 standard deviations at m_f 0.25 and 4.9 at 0.1, and an unknown rate sensitivity costs a lot at high friction. So: measure the diameter, and define exactly where. |
| 1:20-1:50 | `results/validation/figures/ring_test_pop2023.png` | First contact with experiment: published AA6068 ring tests (Pop et al. 2023), three lubricants at 20, 30 and 40% reduction. One friction factor per lubricant reproduces the measured inner diameters at all three reductions to within 0.02 to 0.03, in the same order as the authors' estimates. A consistency check against one lab, not a validation: the alloy's flow curve and rate sensitivity are unknown. |
| 1:50-2:30 | Streamlit dashboard (drift run), then `results/plant/experiments.json` | The surrogate runs at a simulated press: OPC-UA, an edge service, go / check / stop. A load-cell offset drift is caught 19 parts after it starts (20 of 20 seeds); accuracy drops from 52 of 60 to 13 of 58 until the service re-zeroes. A true friction drift is not caught by any monitor; that is the next thing to build. |
| 2:30-2:50 | `results/plant/drift/bundles/part_00494/rerun.json` | Every miss writes a repro bundle with a solver case R&D can rerun. Rerunning this one showed the inputs were wrong, not the model. |

## Commands to have ready (copy-paste)
```bash
python -m pytest -q
python -m ringsim.verify --out results/verify                 # about 3 minutes; show the report, not the run
python -m serving.check_model surrogates.gp_bo.api:load artifacts/gp
python -m plant.demo --scenario drift --parts 600 --model surrogates.gp_bo.api:load --artifact artifacts/gp --data full --out results/plant/drift   # about 3 minutes; pre-run, do not run live
python -m plant.rerun_bundle results/plant/drift/bundles/part_00494                                                       # about 15 seconds
```

## If asked
- *Why a ring test?* It is the standard way forging engineers calibrate friction, and the inner diameter barely
  depends on material strength, so it is a clean friction sensor.
- *Why is force a weak friction signal?* Force scales with strength, and strength is uncertain; at higher friction the
  force barely responds to friction at all.
- *Is the GNN useful?* Not on this geometry: a POD-plus-GP field model beats it in distribution. It would need a
  geometry family that a fixed grid cannot represent.
- *Real-time?* No: strokes are published in bursts and the press cycle is compressed 24 times.
- *Speed-up against industrial solvers?* No claim. `ringsim` is teaching-scale.
- *Who built it?* AI agents wrote the code under my direction; the plan, the gates and the choices of what to fix were
  mine, and `docs/explain/` is how I learned each part. Be ready to derive the weak form and explain the posterior.
