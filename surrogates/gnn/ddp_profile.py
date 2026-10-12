"""Task N7: exercise the DDP code path (gloo backend, 2 CPU processes) and profile one epoch.

What this does and does NOT show
  * It proves the data-parallel code path runs: 2 spawned CPU processes, torch.distributed gloo, DDP
    gradient all-reduce, identical weights afterwards. It is NOT a multi-node or GPU/NCCL test.
  * Timings are for a laptop shared with a running simulation campaign; they are a profile of where
    time goes, not a scaling claim.

    python -m surrogates.gnn.ddp_profile --data smoke --out results/gnn
"""
from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from .results import write_result
from .train import Config, compute_losses, fit, launch_ddp, load_config, prepare
from .model import MeshGNN
from torch_geometric.data import Batch


def op_profile(cfg: Config, dataset: str, top: int = 8) -> list[dict]:
    """Top CPU ops over one training epoch of a single process (torch.profiler)."""
    nm, tr, _, _, _ = prepare(cfg, dataset)
    model = MeshGNN(cfg.latent, cfg.steps)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    batches = [Batch.from_data_list(tr[i:i + cfg.batch_size]) for i in range(0, len(tr) - cfg.batch_size + 1, cfg.batch_size)]
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for b in batches:
            l = compute_losses(model, b, nm, cfg)
            opt.zero_grad(); l["total"].backward(); opt.step()
    evs = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:top]
    tot = sum(e.self_cpu_time_total for e in prof.key_averages())
    return [{"op": e.key, "self_cpu_ms": e.self_cpu_time_total / 1e3, "calls": e.count,
             "share": e.self_cpu_time_total / tot} for e in evs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--config", default="surrogates/gnn/configs/base.yaml")
    ap.add_argument("--out", default="results/gnn")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per process")
    a = ap.parse_args(argv)
    cfg = replace(load_config(a.config), epochs=2, val_frac=0.0, threads=a.threads)
    with tempfile.TemporaryDirectory() as td:
        single: dict = {}
        fit(cfg, a.data, Path(td) / "single", resume=False, profile=single)
        rows = launch_ddp(cfg, a.data, Path(td) / "ddp", world=2)
    ddp = {f"rank{r}": prof for r, _, prof in rows}
    ops = op_profile(cfg, a.data)
    metrics = {
        "single_process_epoch": single,
        "ddp_gloo_2_cpu_procs_epoch": ddp,
        "ddp_epoch_s_max_rank": max(p["epoch_s"] for p in ddp.values()),
        "epoch_time_ratio_noisy": single["epoch_s"] / max(p["epoch_s"] for p in ddp.values()),
        "replica_weights_identical": abs(ddp["rank0"]["param_checksum"] - ddp["rank1"]["param_checksum"])
        <= 1e-9 * abs(ddp["rank0"]["param_checksum"]),
        "top_cpu_ops_single_process": ops,
        "model": {"latent": cfg.latent, "steps": cfg.steps, "batch_size": cfg.batch_size, "threads_per_proc": cfg.threads},
        "platform": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                     "backend": "gloo", "init_method": "file://", "device": "cpu"},
    }
    cmd = "python -m surrogates.gnn.ddp_profile " + " ".join(sys.argv[1:])
    p = write_result(f"{a.out}/n7_ddp_profile_{a.data}.json", "N7", "DDP (gloo, 2 CPU processes) smoke test and one-epoch profile",
                     cmd, {"dataset": a.data, "n_runs": None}, metrics,
                     notes="Timings were taken on a loaded machine and show NO speed-up claim (epoch_time_ratio_noisy is noise). Profile of the 2nd epoch (after warm-up). Machine was shared with a running campaign, so "
                           "absolute timings are noisy. With graphs this small, per-process overhead and gradient "
                           "all-reduce can outweigh the gain; this checks correctness of the code path, not scaling.")
    print(p)
    print({k: v for k, v in metrics.items() if k.startswith(("epoch_time_ratio", "replica", "ddp_epoch"))})


if __name__ == "__main__":
    main()
