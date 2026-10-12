"""Train the mesh GNN (tasks N3, N4, N7).

    python -m surrogates.gnn.train --config surrogates/gnn/configs/base.yaml --data smoke --out artifacts/gnn/base
    python -m surrogates.gnn.train ... --ddp-procs 2        # gloo, CPU, data-parallel code path

Design notes
  * Resumable: `last.pt` (model + optimiser + epoch) is written every epoch; rerunning the same command
    continues from it. `log.jsonl` is append-only, one JSON line per epoch.
  * Split by run: validation runs are held out of the `train` split by run id, never by step.
  * Loss = MSE(disp) + MSE(strain) + MSE(force) + lambda_j * mean_elements (J-1)^2, all targets
    normalised separately. J uses the *denormalised* predicted displacement (physical units).
  * fp32 throughout; torch threads limited by `threads` so a background run leaves cores free.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Batch

from .graph import TEST_SPLITS, Normaliser, assert_no_case_overlap, RunGraph, load_graphs, make_samples, split_run_ids
from .model import MeshGNN
from .physics import j_torch


@dataclass
class Config:
    name: str = "base"
    seed: int = 0
    latent: int = 64
    steps: int = 10               # message-passing blocks (10-15 per the plan)
    lr: float = 1e-3
    lr_final_frac: float = 0.05   # cosine decay to lr * this
    epochs: int = 200
    batch_size: int = 16
    val_frac: float = 0.15
    lambda_j: float = 0.0         # 0 = "base", > 0 = "jloss"
    j_loss: str = "mse"           # mse | huber (huber is robust to the solver's own J outliers)
    w_disp: float = 1.0
    w_strain: float = 1.0
    w_force: float = 1.0
    lambda_id: float = 0.0        # > 0: extra loss on the inner-diameter read-off (percent points / id_scale)
    id_scale: float = 5.0         # percent points; makes the ID term O(1) when the error is a few points
    extra_train: str = ""         # optional second dataset whose `train` split is mixed into training
    extra_val_frac: float = 0.15  # fraction of the extra runs held out for validation
    threads: int = 4
    overfit_samples: int = 0      # > 0: train on the first k samples only (N3 sanity check)
    ckpt_every: int = 1


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    raw = yaml.safe_load(open(path)) or {}
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    known = {f.name for f in fields(Config)}
    bad = set(raw) - known
    if bad:
        raise ValueError(f"unknown config keys: {sorted(bad)}")
    return Config(**raw)


# ----------------------------------------------------------------------------- loss / eval
def compute_losses(model, batch, nm: Normaliser, cfg: Config) -> dict:
    d, s, f = model(batch)
    l_disp = torch.mean((d - batch.y_disp) ** 2)
    l_strain = torch.mean((s - batch.y_strain) ** 2)
    l_force = torch.mean((f - batch.y_force) ** 2)
    pos = batch.p0 + nm.denorm("disp", d)
    J = j_torch(pos, batch.p0, batch.tri_index)
    if cfg.j_loss == "huber":
        l_j = torch.nn.functional.huber_loss(J, torch.ones_like(J), delta=0.1)
    else:
        l_j = torch.mean((J - 1.0) ** 2)
    pm = batch.probe > 0.5                          # the inner mid-plane node of each graph
    id_err = 100.0 * (nm.denorm("disp", d)[pm][:, 0] - nm.denorm("disp", batch.y_disp)[pm][:, 0]) / batch.r0n[pm]
    l_id = torch.mean((id_err / cfg.id_scale) ** 2)  # inner-diameter error in percentage points
    total = (cfg.w_disp * l_disp + cfg.w_strain * l_strain + cfg.w_force * l_force + cfg.lambda_j * l_j
             + cfg.lambda_id * l_id)
    return {"total": total, "disp": l_disp, "strain": l_strain, "force": l_force, "j": l_j, "id": l_id,
            "abs_j_minus_1": (J - 1.0).abs().mean()}


@torch.no_grad()
def evaluate_samples(model, samples, nm, cfg, bs: int = 64) -> dict:
    model.eval()
    acc, n = {}, 0
    for i in range(0, len(samples), bs):
        b = Batch.from_data_list(samples[i:i + bs])
        l = compute_losses(model, b, nm, cfg)
        w = len(samples[i:i + bs])
        for k, v in l.items():
            acc[k] = acc.get(k, 0.0) + float(v) * w
        n += w
    return {k: v / max(n, 1) for k, v in acc.items()}


# ----------------------------------------------------------------------------- data
def prepare(cfg: Config, dataset: str):
    train_ids, val_ids = split_run_ids(dataset, cfg.val_frac, cfg.seed)
    gtr = load_graphs(dataset, "train", train_ids)
    gva = load_graphs(dataset, "train", val_ids) if val_ids else []
    if cfg.extra_train:                      # mix in train runs of a second dataset (e.g. finer mesh density)
        assert_no_case_overlap(cfg.extra_train, "train", dataset, TEST_SPLITS)
        e_tr, e_va = split_run_ids(cfg.extra_train, cfg.extra_val_frac, cfg.seed)
        gtr += load_graphs(cfg.extra_train, "train", e_tr)
        gva += load_graphs(cfg.extra_train, "train", e_va) if e_va else []
    nm = Normaliser.fit(gtr)
    tr = make_samples(gtr, nm)
    if cfg.overfit_samples:
        tr = tr[:: max(len(tr) // cfg.overfit_samples, 1)][: cfg.overfit_samples]   # spread across runs
    va = make_samples(gva, nm)
    return nm, tr, va, train_ids, val_ids


# ----------------------------------------------------------------------------- fit
def fit(cfg: Config, dataset: str, out: Path, rank: int = 0, world: int = 1, resume: bool = True,
        profile: dict | None = None) -> dict:
    """Train; returns the last epoch's log row. `profile` (optional dict) collects per-phase timings."""
    torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)   # same init on all ranks
    out = Path(out)
    nm, tr, va, train_ids, val_ids = prepare(cfg, dataset)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        nm.save(out / "norm.json")
        (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))
        (out / "split.json").write_text(json.dumps({"train_runs": train_ids, "val_runs": val_ids,
                                                    "dataset": dataset}, indent=1))
    model = MeshGNN(cfg.latent, cfg.steps)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start = 0
    ck = out / "last.pt"
    if resume and ck.exists():
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["epoch"]
        best = st.get("best", float("inf"))
    else:
        best = float("inf")
    net = model
    if world > 1:
        net = torch.nn.parallel.DistributedDataParallel(model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1),
                                                       eta_min=cfg.lr * cfg.lr_final_frac)
    import warnings
    with warnings.catch_warnings():                  # fast-forward on resume; stepping before opt.step is intended
        warnings.simplefilter("ignore")
        for _ in range(start):
            sched.step()
    last = {}
    nb = max(len(tr) // cfg.batch_size, 1)          # equal batch count per epoch on every rank
    for epoch in range(start, cfg.epochs):
        t0 = time.perf_counter()
        net.train()
        g = torch.Generator().manual_seed(cfg.seed * 100003 + epoch)   # same permutation on all ranks
        perm = torch.randperm(len(tr), generator=g).tolist()
        mine = perm[rank::world]
        nb_r = max(len(tr) // (cfg.batch_size * world), 1) if world > 1 else nb
        acc, tdata, tfwd, tbwd = {}, 0.0, 0.0, 0.0
        for bi in range(nb_r):
            ts = time.perf_counter()
            idx = mine[bi * cfg.batch_size:(bi + 1) * cfg.batch_size] or mine[:cfg.batch_size]
            b = Batch.from_data_list([tr[i] for i in idx])
            t1 = time.perf_counter()
            l = compute_losses(net, b, nm, cfg)
            t2 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            l["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            t3 = time.perf_counter()
            tdata += t1 - ts; tfwd += t2 - t1; tbwd += t3 - t2
            for k, v in l.items():
                acc[k] = acc.get(k, 0.0) + float(v.detach() if hasattr(v, "detach") else v)
        sched.step()
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "time_s": time.perf_counter() - t0,
               "world": world, **{f"train_{k}": v / nb_r for k, v in acc.items()}}
        if profile is not None and epoch == min(start + 1, cfg.epochs - 1):   # 2nd epoch: skip warm-up
            profile.update({"data_collate_s": tdata, "forward_s": tfwd, "backward_opt_s": tbwd,
                            "epoch_s": row["time_s"], "batches": nb_r, "rank": rank, "world": world})
        if rank == 0:
            if va:
                row.update({f"val_{k}": v for k, v in evaluate_samples(model, va, nm, cfg).items()})
            score = row.get("val_total", row["train_total"])
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch + 1}, out / "best.pt")
            if (epoch + 1) % cfg.ckpt_every == 0 or epoch + 1 == cfg.epochs:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1,
                            "best": best, "cfg": asdict(cfg)}, ck)
            with open(out / "log.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if (epoch + 1) % 10 == 0 or epoch == start:
                print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if world > 1:
            torch.distributed.barrier()
        last = row
    if profile is not None:
        profile["param_checksum"] = float(sum(p.double().abs().sum() for p in model.parameters()))
    return last


# ----------------------------------------------------------------------------- DDP launcher (gloo, CPU)
def _ddp_worker(rank, world, cfg_dict, dataset, out, init_file, q):
    import torch.distributed as dist
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    try:
        cfg = Config(**cfg_dict)
        prof: dict = {}
        row = fit(cfg, dataset, Path(out), rank=rank, world=world, resume=False, profile=prof)
        q.put((rank, row, prof))
    finally:
        dist.destroy_process_group()


def launch_ddp(cfg: Config, dataset: str, out: Path, world: int = 2) -> list:
    """Spawn `world` CPU processes with the gloo backend; returns [(rank, last_row, profile)]."""
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as td:
        init_file = (Path(td) / "rendezvous").as_posix()
        procs = [ctx.Process(target=_ddp_worker, args=(r, world, asdict(cfg), dataset, str(out), init_file, q))
                 for r in range(world)]
        for p in procs:
            p.start()
        res = [q.get(timeout=1800) for _ in procs]
        for p in procs:
            p.join()
        if any(p.exitcode for p in procs):
            raise RuntimeError("a DDP worker failed")
    return sorted(res, key=lambda x: x[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="smoke")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--overfit-samples", type=int)
    ap.add_argument("--ddp-procs", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing last.pt")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"epochs": a.epochs, "threads": a.threads, "overfit_samples": a.overfit_samples})
    if a.ddp_procs > 1:
        print(launch_ddp(cfg, a.data, Path(a.out), a.ddp_procs))
    else:
        fit(cfg, a.data, Path(a.out), resume=not a.fresh)


if __name__ == "__main__":
    main()
