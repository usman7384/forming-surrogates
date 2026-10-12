"""Press simulator: replays held-out solver runs as forged parts and degrades the signals like a real line would.

What is real and what is made up
- Real (from the FE solver): the inner-diameter change and the die-force curve of each replayed run.
- Made up here: plant-scale units (ID 30 mm, K = 100 MPa s^m, die speed 5 mm/s), a lot structure, sensor noise,
  timing jitter, dropped samples, force-offset drift, the die thermocouple (a NUISANCE channel: the solver is
  isothermal, so temperature has no physical effect on any part), and the delayed gauge labels.

Determinism: part i is generated from `default_rng([seed, i])` (lots from `[seed, 1_000_000 + lot]`), so the
same seed gives bit-identical parts regardless of the order in which they are requested.

Lot = lubricant batch: every `lot_len` parts the simulator switches to a different replayed run, i.e. a new ring
geometry and a new friction factor. Friction is therefore constant within a lot, which is what lets the edge
service learn it from a few delayed inner-diameter measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.data import Run, iter_runs
from .config import PlantConfig, drift_type
from .spec import load_spec

V_APPROACH_MM_S = 10.0        # die speed during approach and retract
HOLD_S = 0.05
PAD_S = 0.05
ID_REF_RADIUS_UNITS = 1.0     # solver lengths are in units of the inner radius


@dataclass
class Part:
    part_id: int
    lot_id: int
    ticket: dict
    streams: dict                     # name -> (t, v, q) arrays after degradation
    t_start: float
    t_end: float
    truth: dict = field(default_factory=dict)   # simulation-only ground truth; never published on OPC-UA


def load_pool(cfg: PlantConfig) -> list[Run]:
    """Held-out runs that reach the nominal reduction with some margin (stopped-early runs are skipped)."""
    pool = []
    for sp in cfg.splits:
        for run in iter_runs(cfg.dataset, sp):
            if run.max_reduction >= load_spec().process["nominal_reduction"] + 0.03:
                pool.append(run)
    if not pool:
        raise RuntimeError(f"no usable replay runs in {cfg.dataset} splits {cfg.splits}")
    return pool


class PressSimulator:
    def __init__(self, cfg: PlantConfig, pool: list[Run] | None = None):
        self.cfg = cfg
        self.proc = load_spec().process
        self.pool = pool if pool is not None else load_pool(cfg)
        self.drift_type = drift_type(cfg.scenario)
        self.drift_start = int(round(cfg.drift_start_frac * cfg.n_parts)) if self.drift_type else None
        self._lot_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------ lots
    def lot(self, lot_id: int) -> dict:
        if lot_id not in self._lot_cache:
            rng = np.random.default_rng([self.cfg.seed, 1_000_000 + lot_id])
            cands = self._lot_candidates(lot_id)
            run = cands[int(rng.integers(len(cands)))]
            self._lot_cache[lot_id] = dict(
                run=run, K=self.proc["flow_stress_K_nominal_MPa_s_m"] * float(np.exp(rng.normal(0, self.cfg.k_lot_sd))),
                target=float(run.at([self.proc["nominal_reduction"]], "id_change_pct")[0]))
        return self._lot_cache[lot_id]

    def _lot_candidates(self, lot_id: int) -> list:
        """Runs a lot may replay. Friction drift: lots started at or after the drift start only draw from runs with
        higher and higher true friction (the geometry changes with the run too: the pool has no run pairs that
        differ in friction alone, so this drift is confounded with lot geometry and says so in the docs)."""
        if self.drift_type != "friction" or lot_id * self.cfg.lot_len < self.drift_start:
            return self.pool
        first = -(-self.drift_start // self.cfg.lot_len)                      # first lot that starts after the drift start
        n_lots = max(1, -(-self.cfg.n_parts // self.cfg.lot_len) - first)
        j = max(0, lot_id - first)
        q = self.cfg.friction_drift_q0 + (self.cfg.friction_drift_q1 - self.cfg.friction_drift_q0) * j / max(1, n_lots - 1)
        mfs = np.array([r.case["mf"] for r in self.pool])
        keep = [r for r, m in zip(self.pool, mfs) if m >= np.quantile(mfs, q)]
        return keep or self.pool

    def force_offset_kN(self, i: int) -> float:
        if self.drift_type != "offset" or i < self.drift_start:
            return 0.0
        return self.cfg.drift_slope_kN_per_part * (i - self.drift_start + 1)

    def force_gain(self, i: int) -> float:
        """Multiplicative load-cell gain (1 = healthy). Ramps linearly from the drift start."""
        if self.drift_type != "gain" or i < self.drift_start:
            return 1.0
        return 1.0 + self.cfg.drift_gain_per_part * (i - self.drift_start + 1)

    # ------------------------------------------------------------------ parts
    def _scalar_truth(self, i: int, rng=None) -> tuple[float, float]:
        """Achieved reduction and true ID change of part i. First two draws of the part's rng, so it is cheap
        to recompute (labels need it without regenerating the signals)."""
        rng = rng if rng is not None else np.random.default_rng([self.cfg.seed, i])
        run: Run = self.lot(i // self.cfg.lot_len)["run"]
        r = float(np.clip(rng.normal(self.proc["nominal_reduction"], self.cfg.reduction_sd), 0.05, run.max_reduction - 0.005))
        return r, float(run.at([r], "id_change_pct")[0] + rng.normal(0, self.cfg.id_scatter_pct))

    def part(self, i: int) -> Part:
        cfg, P = self.cfg, self.proc
        rng = np.random.default_rng([cfg.seed, i])
        lot_id = i // cfg.lot_len
        lot = self.lot(lot_id)
        run: Run = lot["run"]
        c = run.case
        r_true, id_true = self._scalar_truth(i, rng)

        R_mm = P["inner_diameter_mm"] / 2
        h0_mm = c["h_id"] * R_mm                       # half height (solver: h0 = h_id in inner-radius units)
        A0_mm2 = np.pi * ((c["od_id"] * R_mm) ** 2 - R_mm ** 2)
        V, A = P["die_speed_mm_s"], P["approach_distance_mm"]
        travel = r_true * h0_mm

        # piecewise-linear die trajectory (line clock relative to part start)
        t_app = A / V_APPROACH_MM_S
        t_form = travel / V
        t_ret = (A + travel) / V_APPROACH_MM_S
        knots_t = np.array([0.0, t_app, t_app + t_form, t_app + t_form + HOLD_S, t_app + t_form + HOLD_S + t_ret])
        knots_p = np.array([0.0, A, A + travel, A + travel, 0.0])
        dur = float(knots_t[-1] + PAD_S)

        xr = np.concatenate([[0.0], run.curves["reduction"]])
        fn = np.concatenate([[0.0], run.curves["force_norm"]])
        scale_kN = lot["K"] * (V / h0_mm) ** c["m"] * A0_mm2 / 1000.0

        def pos_true(t):
            return np.interp(t, knots_t, knots_p)

        def force_true(t):
            p = pos_true(t)
            loaded = (t >= t_app) & (t <= knots_t[3])
            red = np.clip((p - A) / h0_mm, 0.0, None)
            return np.where(loaded, np.interp(red, xr, fn) * scale_kN, 0.0)

        t0 = i * cfg.assumed_cycle_time_s
        offset, gain = self.force_offset_kN(i), self.force_gain(i)
        streams = {}
        for name in ("stroke_position", "stroke_force", "die_temperature"):
            sig = load_spec().signals[name].raw
            rate, jit_ms, drop = sig["sampling_rate_hz"], sig["timing"]["jitter_sd_ms"], sig["timing"]["drop_probability"]
            n = int(dur * rate)
            tn = np.arange(n) / rate
            ts = np.clip(tn + rng.normal(0, jit_ms / 1000.0, n), 0.0, dur)   # sampling instants (jittered)
            order = np.argsort(ts)
            ts = ts[order]
            if name == "stroke_position":
                v = pos_true(ts) + rng.normal(0, 0.005, n)
            elif name == "stroke_force":
                v = gain * force_true(ts) + offset + rng.normal(0, cfg.force_noise_kN, n)
            else:
                v = 250.0 + 5.0 * np.sin(2 * np.pi * (t0 + ts) / 600.0) + np.cumsum(rng.normal(0, 0.05, n))
            q = np.zeros(n)
            u = rng.random(n)
            q[u < 0.01] = 1.0                    # uncertain: value kept, flagged
            bad = u > 0.998
            q[bad] = 2.0                         # bad: value is garbage and must be discarded
            v = np.where(bad, rng.normal(0, 1000.0, n), v)
            keep = rng.random(n) >= drop         # dropped samples vanish entirely (gap in timestamps)
            stamp = ts + (cfg.force_clock_offset_ms / 1000.0 if name == "stroke_force" else 0.0)
            streams[name] = (t0 + stamp[keep], v[keep], q[keep])

        ticket = dict(part_id=i, lot_id=lot_id, od_id=c["od_id"], h_id=c["h_id"], m=c["m"],
                      inner_diameter_mm=P["inner_diameter_mm"], nominal_reduction=P["nominal_reduction"],
                      target_id_change_pct=lot["target"], tol_id_pct=cfg.tol_id_pct)
        truth = dict(id_change_pct=id_true, reduction=r_true, mf=c["mf"], K=lot["K"], run_id=run.run_id,
                     force_offset_kN=offset, force_gain=gain, in_spec=bool(abs(id_true - lot["target"]) <= cfg.tol_id_pct))
        return Part(i, lot_id, ticket, streams, t0, t0 + dur, truth)

    # ------------------------------------------------------------------ delayed labels
    def is_sampled(self, i: int) -> bool:
        return i % self.cfg.label_every == self.cfg.label_every - 1

    def label_for(self, i: int) -> dict | None:
        """The gauge result for part i (if sampled): when it arrives and what it says."""
        if not self.is_sampled(i):
            return None
        rng = np.random.default_rng([self.cfg.seed, 2_000_000 + i])
        lo, hi = self.cfg.label_delay_parts
        due = i + int(rng.integers(lo, hi + 1))
        red_true, id_true = self._scalar_truth(i)
        meas = float(id_true + rng.normal(0, self.cfg.label_sigma_pct))
        # the lab also gauges the part height, so the label carries the achieved reduction (+- 0.002)
        return dict(part_id=i, due_index=due, id_change_pct=meas, sigma_pct=self.cfg.label_sigma_pct,
                    reduction=float(red_true + rng.normal(0, 0.002)))

    def labels_due(self, i: int) -> list[dict]:
        """Labels that become available just before part i's stroke (only parts that were already made)."""
        lo, hi = self.cfg.label_delay_parts
        out = []
        for j in range(max(0, i - hi), i):
            lab = self.label_for(j)
            if lab is not None and lab["due_index"] == i:
                out.append(lab)
        return out
