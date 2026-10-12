"""Operator dashboard (task P6).   streamlit run dashboard/app.py -- --results results/plant/drift

Shows what an operator needs per shift: the verdict of the latest part, how the model's accuracy is trending (with
an interval, because labels are few), drift alerts, and how late the labels are. The slider replays the shift up to
a chosen part, so the dashboard can be shown without a live press. It reads files written by plant.demo; it does not
talk to OPC-UA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.data import acceptance_at, acceptance_series, drift_status, label_status, load_results  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOUR = {"go": "#2e7d32", "check": "#f9a825", "stop": "#c62828"}


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/plant/drift")
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_data
def _load(path: str):
    return load_results(path)


def main():
    st.set_page_config(page_title="Press 1 - surrogate monitor", layout="wide")
    args = _args()
    if not (Path(args.results) / "metrics.json").exists():
        st.info(f"Waiting for results in {args.results}: the edge service writes them when the shift ends.")
        st.stop()
    R = _load(args.results)
    m = R.metrics["metrics"]
    cfg_every = m["labels"]["sampled_every"]
    n = len(R.parts)
    st.title("Press 1 - inner-diameter surrogate monitor")
    st.caption(f"SIMULATED press, replayed from {args.results}. Model: {m['model']['name']} v{m['model']['version']}. "
               "Nothing here is validated against a real machine.")
    now = st.slider("Replay the shift up to part", 1, n - 1, n - 1)
    cur = R.parts.iloc[now]
    win = int(m["acceptance"]["window_parts"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest verdict", cur.verdict.upper(), cur.reason.replace("_", " "))
    c1.markdown(f"<div style='height:8px;background:{COLOUR[cur.verdict]}'></div>", unsafe_allow_html=True)
    c2.metric("Predicted ID change", f"{cur.pred_mean:+.2f} %", f"+/- {2 * cur.pred_std:.2f} (2 sd)", delta_color="off")
    k, nl, p, lo, hi = acceptance_at(R.labels, now, win)
    c3.metric(f"Acceptance (last {win} parts)", "n/a" if nl == 0 else f"{p:.0%}",
              f"{k}/{nl} measured; Wilson 95% [{lo:.0%}, {hi:.0%}]", delta_color="off")
    ds = drift_status(R.events, now)
    c4.metric("Drift alert (load-cell zero)", "ACTIVE" if ds["active"] else "none",
              f"first at part {ds['first']}" if ds["first"] is not None else "no alert yet", delta_color="off")
    if ds["active"]:
        st.error("Force-sensor drift alert: the load-cell zero reading has shifted away from its reference window. "
                 "Check the amplifier zero and the contact detection before trusting predictions.")

    left, right = st.columns(2)
    with left:
        st.subheader("Load-cell zero reading (drift feature)")
        fig, ax = plt.subplots(figsize=(6, 2.8), dpi=120)
        d = R.parts.iloc[:now + 1]
        ax.plot(d.part_id, d.f_zero_kN, lw=0.8)
        for e in R.events:
            if e["type"] == "drift_alert" and e["primary"] and e["part_index"] <= now:
                ax.axvline(e["part_index"], color="r", alpha=0.15)
        ax.set_xlabel("part number")
        ax.set_ylabel("zero reading (kN)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Acceptance trend with Wilson 95% interval")
        s = acceptance_series(R.labels[R.labels.part_index <= now] if not R.labels.empty else R.labels, now + 1, win)
        if len(s):
            fig, ax = plt.subplots(figsize=(6, 2.8), dpi=120)
            ax.plot(s.part, s.rate)
            ax.fill_between(s.part, s.low, s.high, alpha=0.25)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("part number")
            ax.set_ylabel("share within tolerance")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough measured parts yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Verdicts per 50 parts")
        d = R.parts.iloc[:now + 1].copy()
        d["block"] = (d.part_id // 50) * 50
        st.bar_chart(d.groupby(["block", "verdict"]).size().unstack(fill_value=0))
    with right:
        st.subheader("Labels: late and sparse")
        ls = label_status(R.parts, R.labels, now, cfg_every)
        st.write(f"Only 1 part in {cfg_every} is measured. So far {ls['sampled']} parts were sampled, "
                 f"{ls['measured']} measurements have arrived, {ls['waiting']} are still waiting "
                 f"(parts {ls['waiting_parts']}). {ls['unsampled_share']:.0%} of parts will never be measured.")
        if not R.labels.empty:
            lab = R.labels[R.labels.part_index <= now]
            st.write(f"Label delay: median {lab.label_delay_parts.median():.0f} parts "
                     f"(range {lab.label_delay_parts.min()}-{lab.label_delay_parts.max()}).")

    st.subheader("Recent parts")
    cols = ["part_id", "lot_id", "verdict", "reason", "pred_mean", "pred_std", "mf_mean", "mf_std", "reduction_est",
            "f_zero_kN", "measured", "latency_ms"]
    st.dataframe(R.parts.iloc[max(0, now - 29):now + 1][cols].iloc[::-1], hide_index=True)

    st.subheader("Repro bundles")
    shown = [b for b in R.bundles if b["part_id"] <= now]
    if not shown:
        st.write("No bundles yet.")
    for b in shown[-5:]:
        with st.expander(f"part {b['part_id']}: {b['reason']} (residual {b['residual']:+.2f} points)"):
            st.code(b["rerun"])
            png = Path(b["path"]) / "plot.png"
            if png.exists():
                st.image(str(png))
    lat = m["latency"]
    st.caption(f"Latency: assumed cycle time {lat['assumed_cycle_time_s']} s; stroke end to verdict "
               f"p50 {lat['end_to_end_ms_p50']:.1f} ms, p99 {lat['end_to_end_ms_p99']:.1f} ms (localhost, simulated).")


main()
