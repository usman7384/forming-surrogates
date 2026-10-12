# The mesh GNN, explained for an ML engineer new to CAE

Status: trained and evaluated on the full dataset (360 train runs; tests: 40 random, 40 out-of-distribution
geometry, 20 fine mesh), 30 epochs per model. Sections about smoke data only show the pipeline works. Nothing
here is validated against experiments. Numbers are in the results section below and come only from
`results/gnn/*.json`.

## Results on the full data (read this first)

**The 30-epoch cap is a compute-budget choice, not a tuned value.** An epoch took about 200-400 s on a shared
laptop CPU (see `artifacts/gnn/*/log.jsonl`), so the planned 300 epochs (about a day and a half) was cut to 30.
The models were trained once with one seed, so differences between runs smaller than a few percent are not
distinguishable from noise (noise was not measured). Files: `results/gnn/eval_base_full/metrics.json`,
`results/gnn/eval_jloss_full/metrics.json`, `results/gnn/n4_base_vs_jloss_full.json`,
`results/gnn/n5_mlp_baseline_full.json`. Failure-gallery figures are in each `eval_*_full/figures/`.

Field error is normalised RMSE (RMSE divided by the standard deviation of the true values; 0 is perfect, 1 is
no better than the mean). Models: `base` is lambda_j = 0, `jloss` is lambda_j = 1.

| Split (runs) | model | nRMSE dr | nRMSE dz | nRMSE strain | inner-diameter RMSE (% points) | mean abs(J-1) predicted (solver) |
| --- | --- | --- | --- | --- | --- | --- |
| test_random (40) | base | 0.061 | 0.030 | 0.096 | 1.20 | 0.034 (0.019) |
| test_random (40) | jloss | 0.053 | 0.035 | 0.088 | 1.26 | 0.021 (0.019) |
| test_ood_geometry (40) | base | 0.135 | 0.065 | 0.181 | 3.32 | 0.029 (0.016) |
| test_ood_geometry (40) | jloss | 0.131 | 0.063 | 0.164 | 2.79 | 0.022 (0.016) |
| test_fine_mesh (20) | base | 0.396 | 0.257 | 0.583 | 5.43 | 0.054 (0.014) |
| test_fine_mesh (20) | jloss | 0.403 | 0.234 | 0.560 | 5.38 | 0.042 (0.014) |

### A field model without a graph: POD + GP (`results/gnn/pod_field_baseline_full.json`)

To compare the GNN with something that also predicts fields, `surrogates/baselines/pod_field.py` resamples every
run's fields onto one fixed grid in the initial normalised (r, z) coordinates, keeps 24 POD modes (99.97% of the
energy) and fits a GP from (od_id, h_id, m_f, m, reduction) to the mode coefficients. Same nRMSE definition and
same rows as the GNN.

| Split | model | nRMSE dr | nRMSE dz | nRMSE strain | inner-diameter RMSE (% points) |
| --- | --- | --- | --- | --- | --- |
| test_random | POD+GP | 0.020 | 0.019 | 0.056 | 0.22 |
| test_random | GNN jloss | 0.053 | 0.035 | 0.088 | 1.26 |
| test_ood_geometry | POD+GP | 0.214 | 0.252 | 0.245 | 3.51 |
| test_ood_geometry | GNN jloss | 0.131 | 0.063 | 0.164 | 2.79 |
| test_fine_mesh | POD+GP | 0.020 | 0.021 | 0.147 | 0.28 |
| test_fine_mesh | GNN jloss | 0.403 | 0.234 | 0.560 | 5.38 |

Said plainly: **in distribution this field model beats the GNN by 1.6 to 3x (3x for displacement dr, 1.6 to 1.7x for dz and strain), and the GNN's field value is not shown
there.** The fine-mesh gap is partly structural: the POD+GP lives on a mesh-independent grid, so a finer mesh costs it
only an interpolation error (the "resampling floor" entries in the JSON are 0.005 in dr), whereas the GNN has to
generalise to new edge lengths. The GNN wins only on the out-of-distribution geometry split (dr 0.13 vs 0.21). On
test_random the POD+GP error is close to its own POD truncation floor (dr 0.013, in the JSON), so more modelling
effort there would mostly buy nothing. Its inner-diameter error (0.22) is the same order as the scalar MLP (0.10).

### Two follow-up experiments (all in `results/gnn/variants_comparison_full.json`)

Both are single runs (one seed, 30 epochs, a compute-budget cap). Seed-to-seed spread was **not measured**, so
differences of a few percent are not established. Extra seeds were not trained.

**Does an inner-diameter auxiliary loss close the gap? No.** `idloss` is `jloss` plus a loss on the predicted
inner mid-plane node against the solver's (in percentage points). Inner-diameter RMSE got slightly worse, not better:

| Split | jloss | idloss | scalar MLP | POD+GP |
| --- | --- | --- | --- | --- |
| test_random | 1.26 | 1.52 | 0.10 | 0.22 |
| test_ood_geometry | 2.79 | 3.91 | 0.71 | 3.51 |
| test_fine_mesh | 5.38 | 5.74 | 0.12 | 0.28 |

Field errors also got worse on the first two splits (test_random dr 0.068 vs 0.053). A hypothesis (untested): the
extra term competes with the displacement term for the same probe node, and 30 epochs is too few to balance them.
The gap to the scalar models (about ten times in distribution) does not shrink with this loss.

**Is the fine-mesh failure a mesh-density shift? Mostly yes.** `mixmesh` is `jloss` with 60 extra training runs on the
finer mesh (nz = 12) from `fine_train`, mixed with the normal training split; no `idloss`, because it did not help.
Leakage check on the real data (`results/gnn/mixmesh_leakage_check.json`): none of the 60 cases equals a test case of
any split (exact match of od_id, h_id, m_f, m within 1e-9: 0 / 0 / 0), and the nearest test case is 0.54 standard
deviations away; the extra set's own `test_random` split (6 runs) was not used. Result on `test_fine_mesh`:

| Model | nRMSE dr | nRMSE dz | nRMSE strain | inner-diameter RMSE (% points) | force mean rel. error |
| --- | --- | --- | --- | --- | --- |
| jloss | 0.403 | 0.234 | 0.560 | 5.38 | 0.038 |
| mixmesh | 0.146 | 0.066 | 0.189 | 3.15 | 0.011 |
| POD+GP | 0.020 | 0.021 | 0.147 | 0.28 | not modelled |

So adding fine-mesh runs cut the fine-mesh field error by 60-70% and it moved to 0.19 on strain, which supports the
reading that most of the failure was a distribution shift in mesh density, not a limit of the architecture. It is not
fully closed: dr is still about seven times the POD+GP value and inner-diameter error is still 3.15 points, so
something besides density (or too few fine runs and epochs) remains; I did not test which. In-distribution
`mixmesh` is no worse (test_random dr 0.048, strain 0.086), though that is one seed and the 60 extra runs also add data.
On `test_ood_geometry` it matches jloss (dr 0.136 vs 0.130).

What this says:

- **Base versus J loss.** The J loss lowered the predicted-J deviation on every split (0.034 to 0.021 on
  test_random), moving it toward the solver's own level, and gave lower strain and out-of-distribution errors.
  It was not better everywhere: on test_random it was worse for dz (0.035 vs 0.030) and the inner-diameter
  error (1.26 vs 1.20), and on fine mesh dr is slightly worse. Mechanically counting all 63 split-by-group-by-metric
  comparisons in `n4_base_vs_jloss_full.json`, jloss is lower in 47 and base in 16; these comparisons overlap (all / early-stopped / completed groups share runs), so they are not independent. With one seed and 30 epochs,
  read this as "a mild improvement in most metrics", not a proven effect.
- **The GNN does not beat the simple MLP baseline on inner diameter.** On the same inner-diameter quantity
  (both read at reductions 0.1, 0.2, 0.3 on the same rows: 115, 112 and 53 for the three test splits; the GNN reads the
  displacement of the inner mid-plane node, the MLP is trained on the quantity directly), the 5-member MLP on scalar inputs has RMSE 0.10 (test_random), 0.71
  (out-of-distribution) and 0.12 (fine mesh) percentage points, versus 1.2, 2.8-3.3 and 5.4 for the GNN. For
  the quantity the project cares most about, the GNN is roughly ten times worse than the MLP in-distribution (single seed, 30 epochs, different model types). The
  MLP gives no field, so the GNN's value, if any, is the full displacement and strain field, not the scalar.
- **Fine mesh is a failure.** Errors are about 4-9 times those of test_random (base / jloss ratios: dr 6.5 / 7.6, dz 8.6 / 6.7, strain 6.1 / 6.3, inner-diameter RMSE 4.5 / 4.3; strain nRMSE itself 0.58 / 0.56). All ten
  gallery cases for both models are fine-mesh runs (run ids 0440-0459); six or seven of them are at s >= 0.94, the rest at s 0.54-0.86. Hypothesis (untested): the edge lengths and node counts are
  outside what training saw, and 30 epochs on one mesh density do not generalise.
- **Early-stopped (fold-over) runs** were kept in training and test, with their steps up to the stop, and are
  reported separately (`early_stopped_runs` / `completed_runs` in every split of the metrics files). They are
  harder out of distribution: out-of-distribution dr nRMSE 0.228 (9 early-stopped runs) versus 0.097 (31 completed)
  for base, and inner-diameter RMSE 6.3 versus 2.2 points. In-distribution the gap is smaller (dr 0.087 vs 0.052).
  On fine mesh the early-stopped group is not worse than the completed one (dr 0.32 vs 0.46), which I cannot
  explain from these data. Group sizes are 9-31 runs, so these splits are noisy.
- **Force** is predicted well in relative terms: mean relative error about 1% on test_random and out-of-distribution,
  and 4-5% on fine mesh.
- **Inference time** was 18.5 ms per stroke position (base, batched per run; 11 to 15 ms for the other variants; timed on a shared laptop) against a mean 63.5 s per solver run;
  not a like-for-like comparison (one sample is one stroke position; the solver is teaching-scale), so I make no
  speed-up claim.
- **Failure gallery.** Each case has a rule-derived hypothesis marked untested. The ten worst per model are all
  fine-mesh runs (0440-0459); the hypotheses cite near-sticking friction (m_f 0.8-1.0), late stroke and fold-over.

The text below this section was written during Wave 1 on smoke data; where it says "not run" for N4/N6, the
results above supersede it.

## What it is, in one paragraph

The solver turns a ring (shape, friction m_f, rate sensitivity m) into a deforming mesh over a stroke.
The GNN is a learned shortcut: give it the ring's **initial mesh**, the inputs, and a **stroke position s**,
and one forward pass returns where every node ends up, the strain at every node, and the die force. It is
trained on solver runs and judged on held-out runs.

## The design decisions (and why)

| Decision | Why |
| --- | --- |
| **One-shot, stroke-conditioned** (s is an input) | No rolling out 35 time steps, so no error accumulation. Cost: the net must learn the whole path from "undeformed" to "deformed at s" directly. |
| **Lagrangian, graph = initial mesh** | The graph never changes during the stroke; the target is displacement = position at s minus initial position. Same topology for every s. |
| **Boundary type only as it was at the start** | Side nodes roll onto the die during the stroke. Contact at s is something the network must predict, not read. Flags are multi-hot (a corner is both "inner" and "on die"); "interior" means no other flag. |
| **Node inputs** | initial (r, z) plus 5 boundary flags; globals m_f, m, s are concatenated to every node. |
| **Edge inputs** | relative position (dr, dz) and length in the initial mesh, edges both ways. |
| **Three outputs** | displacement (2), nodal strain (1; element strain averaged to nodes), and a graph-level normalised force F* from a mean-pool + MLP head. |
| **Each target normalised separately** | Displacements and strains differ in scale by orders of magnitude; statistics are fitted on training runs only and saved in `norm.json`. |
| **Split by run** | Steps of one run are almost the same sample. Validation runs are held out of `train` by run id (`split.json`). |
| **Encode-process-decode, 10-15 blocks** | MeshGraphNets style: MLP encoders, residual message-passing blocks with layer norm, MLP decoders. Many blocks are needed so information can cross the ring. Plain PyTorch Geometric `MessagePassing`. |
| **fp32, CPU** | Graphs are 50-560 nodes. No ONNX export for the GNN (the plant never uses it). |

## The physics loss

Each triangle of an axisymmetric mesh sweeps a ring of volume 2*pi*A*rbar (area times centroid radius). So
the volume ratio is **J = (A/A0)(rbar/rbar0)**, and incompressible metal means J = 1. The optional loss term
is `lambda_j * mean_triangles (J - 1)^2`, computed from the predicted positions (initial position + denormalised
predicted displacement). Signed area is used so a flipped triangle is penalised rather than hidden.

**Measured on the solver's own data first (task N2)**, file `results/gnn/n2_j_distribution_smoke.json`:
over all saved steps of the 13 smoke runs, mean |J-1| is 0.0126, median 0.0037, 99th percentile 0.123, and the
maximum 0.933. So the solver is *mostly* incompressible but has a heavy tail (distorted triangles near the
die corners, worst in the early-stopped fold-over runs). Consequences: (1) the loss has a noise floor and
should not be pushed to exactly 0; (2) `j_loss: huber` is available as a robust variant; (3) the real question
is empirical, so N4 trains `base` (lambda_j = 0) and `jloss` (lambda_j = 1) and the comparison is reported
whichever way it goes. On smoke data this comparison has **not** been run.

## What was built

| Piece | File | Task |
| --- | --- | --- |
| Graph builder, normaliser, run-wise split | `surrogates/gnn/graph.py` | N1 |
| J measurement script | `surrogates/gnn/j_stats.py` | N2 |
| Model | `surrogates/gnn/model.py` | N3 |
| Training (YAML config, resumable, optional DDP) | `surrogates/gnn/train.py`, `surrogates/gnn/configs/*.yaml` | N3, N4, N7 |
| Overfit check | `surrogates/gnn/overfit.py` | N3 |
| MLP baseline (ensemble of 5) | `surrogates/baselines/mlp_scalar.py` | N5 |
| DDP check and profile | `surrogates/gnn/ddp_profile.py` | N7 |
| Evaluation and failure gallery (run on full data, see Results) | `surrogates/gnn/evaluate.py` | N6 |

## How to run

```
# N2: J on solver data
python -m surrogates.gnn.j_stats --data smoke --out results/gnn
# N3: overfit two graphs, then a smoke training run
python -m surrogates.gnn.overfit --data smoke --out results/gnn
python -m surrogates.gnn.train --config surrogates/gnn/configs/smoke.yaml --data smoke --out artifacts/gnn/smoke
# N5: MLP baseline
python -m surrogates.baselines.mlp_scalar --data smoke --out results/gnn
# N7: DDP code path and profile
python -m surrogates.gnn.ddp_profile --data smoke --out results/gnn
# N6 (smoke only until Gate G2)
python -m surrogates.gnn.evaluate --run artifacts/gnn/smoke --data smoke --out results/gnn/smoke_eval
# N4 after Gate G2 (long: run in the background; re-running the same command resumes from last.pt)
python -m surrogates.gnn.train --config surrogates/gnn/configs/base.yaml  --data full --out artifacts/gnn/base
python -m surrogates.gnn.train --config surrogates/gnn/configs/jloss.yaml --data full --out artifacts/gnn/jloss
```

A run directory holds `config.yaml`, `norm.json`, `split.json`, `log.jsonl` (one line per epoch, append-only),
`last.pt` (resume point) and `best.pt` (best validation loss).

## What the smoke results say (and do not say)

- **Overfit (N3)**, `results/gnn/n3_overfit_smoke.json`: two samples from different runs; normalised MSE goes
  from 3.28 to 0.00032 in 400 epochs. The model and loss can memorise; that is all this shows.
- **Smoke training and evaluation** run end to end (`results/gnn/smoke_eval/metrics.json`, a 15-epoch tiny
  model). With 9 training runs, 1-2 runs per test split and a model trained for seconds, those metrics are
  *not* evidence of accuracy, and the failure gallery there has at most four entries (one per test run).
- **MLP baseline (N5)**, `results/gnn/n5_mlp_baseline_smoke.json`: same table as the GP (inner-diameter change
  and log normalised force at reductions 0.1/0.2/0.3). An ensemble of 5 gives mean and spread; its coverage is
  reported as measured and is unreliable with so few rows.

## DDP and profiling (N7)

Command: `python -m surrogates.gnn.ddp_profile`; results `results/gnn/n7_ddp_profile_smoke.json`.

What is verified: **the data-parallel code path runs with the gloo backend on 2 CPU processes** on one
Windows laptop: two spawned processes, DDP gradient all-reduce, and identical weights on both ranks
afterwards (`replica_weights_identical` is true, and a unit test checks it). This is not multi-node, not GPU,
not NCCL, and not a scaling study.

Profile of one epoch (single process, 12 blocks, latent 64): forward about 37% of the epoch, backward plus
optimiser about 63%, batch collation under 1%. The biggest op is dense matrix multiply (`aten::mm`, about
25% of CPU self time), followed by `addmm`, `cat` and scatter-style `index_put`. Message passing therefore
costs mostly in the MLPs on edges, not in the graph bookkeeping. The 2-process and single-process epochs differ by an amount (key `epoch_time_ratio_noisy`) that is not a
speed-up measurement: the laptop was running the simulation campaign at the same time, so timings are
noisy and I do not claim a speed-up from them.

Windows specifics: there is no `fork`, so workers are started with `spawn` (the worker function must be
importable, which it is); rendezvous uses a `file://` store in a temp directory because the default TCP store
depends on libuv support that varies between builds; and `threads` must be lowered per process (default 2 in
the profile), otherwise two processes each grabbing 4 threads oversubscribe the cores and slow both down.
(I saw wildly varying epoch times while the machine was busy; not recorded in a results file, so no number here.)

## What can go wrong

- **Dead-metal zones and high friction.** Near-sticking runs have sharp strain gradients that smooth message
  passing blurs.
- **Extrapolation.** `test_ood_geometry` rings are outside the training box; expect a gap, and it is reported.
- **Fine mesh.** A finer mesh has different node counts and edge lengths; the model sees edge length, but
  edge-length statistics shift. That gap is reported too.
- **Solver outliers.** The heavy tail of |J-1| means some targets are locally noisy; the J loss may fight them.
- **Early-stopped runs** (fold-over) have fewer saved steps; they are kept and flagged, and s is normalised by
  the campaign's target reduction 0.35, so those runs just stop at s < 1.
- **Disk.** Checkpoints go under `artifacts/gnn/`; do not commit files over 50 MB.
- **Not claimed:** no speed-up versus industrial solvers (the solver here is teaching-scale), and no
  validation against experiments.
