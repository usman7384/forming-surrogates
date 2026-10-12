"""Signal cleaning, alignment onto one stroke axis, and feature extraction for one part.

Why this exists: force (1 kHz) and position (250 Hz) arrive at different rates, with jittered timestamps, dropped
samples and bad-quality values. The model needs one number, the achieved height reduction, and the monitoring needs a
few robust force features. Everything here is plain numpy so it is cheap and testable without OPC-UA.

Reduction estimate: the position transducer measures travel from the home position, so it cannot tell where the part
surface is. Contact is found from the force (first sustained crossing of a threshold). This is realistic touch-off
detection, and it is also how a drifting force offset leaks into the prediction (an offset above the threshold makes
the contact appear during the approach).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BAD = 2.0
REF_REDUCTION = 0.20      # reduction at which the reference force is read


def clean(t, v, q) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop bad-quality / non-finite samples, sort by time, remove duplicate stamps. Returns (t, v, n_uncertain)."""
    t, v, q = (np.asarray(a, float) for a in (t, v, q))
    ok = (q < BAD) & np.isfinite(v) & np.isfinite(t)
    t, v, q = t[ok], v[ok], q[ok]
    order = np.argsort(t, kind="stable")
    t, v, q = t[order], v[order], q[order]
    if len(t):
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, v, q = t[keep], v[keep], q[keep]
    return t, v, int((q > 0).sum())


def align(streams: dict) -> dict:
    """Put force and position on the force time axis (linear interpolation of position). Returns arrays t, force, pos
    plus counts. The 3 ms clock offset between the two amplifiers is NOT corrected (it is worth 0.015 mm at 5 mm/s)."""
    tf, f, nf_unc = clean(*streams["stroke_force"])
    tp, p, np_unc = clean(*streams["stroke_position"])
    if len(tf) < 10 or len(tp) < 10:
        raise ValueError("not enough usable samples to align")
    pos = np.interp(tf, tp, p)
    return dict(t=tf, force=f, pos=pos, n_force=len(tf), n_pos=len(tp), n_uncertain=nf_unc + np_unc)


@dataclass
class StrokeFeatures:
    contact_idx: int | None
    contact_pos_mm: float
    end_pos_mm: float
    reduction: float
    f_zero_kN: float          # mean force while the die is still well above the part (load-cell zero check)
    peak_kN: float
    peak_force_norm: float
    contact_anomaly: bool
    n_force: int
    n_pos: int
    force_ref_norm: float = float("nan")   # force at 20% reduction (normalised): independent of the stroke's final reduction


def find_contact(force: np.ndarray, thr: float, debounce: int) -> int | None:
    """First index where `debounce` consecutive samples exceed `thr`."""
    above = force > thr
    if len(above) < debounce:
        return None
    run = np.convolve(above.astype(int), np.ones(debounce, int), mode="valid")
    hit = np.flatnonzero(run == debounce)
    return int(hit[0]) if len(hit) else None


def features(al: dict, ticket: dict, proc: dict, thr_kN: float, debounce: int, anomaly_mm: float = 0.5) -> StrokeFeatures:
    force, pos = al["force"], al["pos"]
    R = ticket["inner_diameter_mm"] / 2
    h0 = ticket["h_id"] * R
    A = proc["approach_distance_mm"]
    idx = find_contact(force, thr_kN, debounce)
    end_pos = float(pos.max())
    contact_pos = float(pos[idx]) if idx is not None else float("nan")
    red = (end_pos - contact_pos) / h0 if idx is not None else float("nan")
    above_part = pos < A - 0.5                              # die still clearly in the air
    f_zero = float(force[above_part].mean()) if above_part.sum() >= 10 else float("nan")
    smooth = np.convolve(force, np.ones(5) / 5, mode="valid")
    peak = float(smooth.max()) if len(smooth) else float(force.max())
    A0 = np.pi * ((ticket["od_id"] * R) ** 2 - R ** 2)
    scale = proc["flow_stress_K_nominal_MPa_s_m"] * (proc["die_speed_mm_s"] / h0) ** ticket["m"] * A0 / 1000.0
    anomaly = idx is None or abs(contact_pos - A) > anomaly_mm
    f_ref = float("nan")
    if True:                                                # (always computed; nan if the stroke did not get that far)
        # force at a fixed die position (the nominal touch-off plus 0.20 of the half height): the peak force depends
        # on how far this particular stroke went (scatter of +-0.008 in reduction), this one does not, and it does
        # not use the force-threshold contact estimate, so it is not moved by a zero offset in the same way
        target = A + REF_REDUCTION * h0
        loading = np.arange(len(pos)) < int(np.argmax(pos))        # forming phase only, not the retract
        near = np.flatnonzero((np.abs(pos - target) <= 0.01 * h0) & loading)
        if len(near) >= 5 and end_pos > target:
            f_ref = float(force[near].mean()) / scale
    return StrokeFeatures(idx, contact_pos, end_pos, red, f_zero, peak, peak / scale, bool(anomaly),
                          al["n_force"], al["n_pos"], f_ref)
