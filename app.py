"""
Battery State of Health and Remaining Useful Life estimator — Streamlit app.

Read a NASA PCoE .mat cell file, or set the features by hand.
SOH:  3-feature linear regression trained on B0005/B0006/B0007/B0018.
RUL:  3-feature random forest trained on B0005/B0006/B0018.
"""

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt

from features import (
    window_features,
    add_rolling_features,
    MIN_CYCLES_FOR_RUL,
)

# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
INK = "#16202B"
MUTED = "#5C6B7A"
RULE = "#C7CDD4"
TRACE = "#0F6E7A"      # measurement trace — electrolyte teal
CAUTION = "#C25E00"    # ageing threshold
CRITICAL = "#A32A28"   # end-of-life threshold

SOH_SCALE_MIN, SOH_SCALE_MAX = 65.0, 100.0

st.set_page_config(page_title="Cell health datasheet", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: {MUTED}; margin-bottom: 0.35rem;
    }}
    .masthead {{
        font-size: 2.1rem; font-weight: 600; color: {INK};
        line-height: 1.15; margin: 0 0 0.4rem 0;
    }}
    .standfirst {{ color: {MUTED}; font-size: 0.95rem; max-width: 62ch; }}
    .hairline {{ border: none; border-top: 1px solid {RULE}; margin: 1.1rem 0; }}

    .specstrip {{ display: flex; gap: 0; flex-wrap: wrap; border-top: 2px solid {INK};
                  border-bottom: 1px solid {RULE}; margin-bottom: 1rem; }}
    .spec {{ flex: 1 1 150px; padding: 0.85rem 1.1rem 0.9rem 0; }}
    .spec + .spec {{ border-left: 1px solid {RULE}; padding-left: 1.1rem; }}
    .spec-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
                   letter-spacing: 0.11em; text-transform: uppercase; color: {MUTED}; }}
    .spec-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.85rem;
                   font-weight: 500; color: {INK}; line-height: 1.3; }}
    .spec-unit {{ font-size: 0.95rem; color: {MUTED}; margin-left: 0.18rem; }}
    .spec-note {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: {MUTED}; }}

    .gauge-wrap {{ margin: 0.2rem 0 0.2rem 0; }}
    .gauge {{ position: relative; height: 34px; background: #E3E7EB;
              border: 1px solid {RULE}; }}
    .gauge-fill {{ position: absolute; top: 0; bottom: 0; left: 0; background: {TRACE};
                   opacity: 0.16; }}
    .gauge-mark {{ position: absolute; top: -6px; bottom: -6px; width: 1px; background: {RULE}; }}
    .gauge-needle {{ position: absolute; top: -7px; bottom: -7px; width: 3px; background: {INK}; }}
    .gauge-axis {{ position: relative; height: 20px; font-family: 'IBM Plex Mono', monospace;
                   font-size: 0.66rem; color: {MUTED}; }}
    .gauge-tick {{ position: absolute; transform: translateX(-50%); padding-top: 3px; }}

    .badge {{ display: inline-block; font-family: 'IBM Plex Mono', monospace;
              font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
              padding: 0.18rem 0.5rem; border: 1px solid currentColor; }}

    .noteblock {{ font-size: 0.86rem; color: {MUTED}; max-width: 78ch; line-height: 1.55; }}
    .noteblock b {{ color: {INK}; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)

mpl.rcParams.update({
    "font.family": "monospace",
    "font.size": 9,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def status_of(soh):
    if soh >= 90:
        return "Healthy", TRACE
    if soh >= 80:
        return "Ageing", CAUTION
    return "Replace soon", CRITICAL


def spec(label, value, unit="", note=""):
    unit_html = f"<span class='spec-unit'>{unit}</span>" if unit else ""
    note_html = f"<div class='spec-note'>{note}</div>" if note else ""
    return (f"<div class='spec'><div class='spec-label'>{label}</div>"
            f"<div class='spec-value'>{value}{unit_html}</div>{note_html}</div>")


def health_gauge(soh, label, colour):
    """Signature element: where this cell sits on the degradation scale."""
    def pos(v):
        v = min(max(v, SOH_SCALE_MIN), SOH_SCALE_MAX)
        return (v - SOH_SCALE_MIN) / (SOH_SCALE_MAX - SOH_SCALE_MIN) * 100

    ticks = "".join(
        f"<div class='gauge-tick' style='left:{pos(v)}%'>{v:.0f}</div>"
        for v in (70, 80, 90, 100)
    )
    marks = "".join(
        f"<div class='gauge-mark' style='left:{pos(v)}%'></div>" for v in (70, 80)
    )
    return f"""
    <div class='gauge-wrap'>
      <div class='gauge'>
        <div class='gauge-fill' style='width:{pos(soh)}%'></div>
        {marks}
        <div class='gauge-needle' style='left:{pos(soh)}%'></div>
      </div>
      <div class='gauge-axis'>{ticks}</div>
      <div style='margin-top:0.55rem'>
        <span class='badge' style='color:{colour}'>{label}</span>
        <span class='spec-note' style='margin-left:0.6rem'>
          Scale reads {SOH_SCALE_MIN:.0f}–{SOH_SCALE_MAX:.0f}% &nbsp;·&nbsp;
          marks at 80% ageing and 70% end of life
        </span>
      </div>
    </div>
    """


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


@st.cache_resource
def load_models():
    return joblib.load("soh_model.pkl"), joblib.load("rul_model.pkl")


soh_bundle, rul_bundle = load_models()
model, FEATURES = soh_bundle["model"], soh_bundle["features"]
rul_model, RUL_FEATURES = rul_bundle["model"], rul_bundle["features"]
RUL_RMSE = rul_bundle.get("cv_rmse_cycles", 10.7)

# ----------------------------------------------------------------------
# Masthead
# ----------------------------------------------------------------------
st.markdown(
    "<div class='eyebrow'>18650 Li-ion cell &nbsp;·&nbsp; NASA PCoE ageing dataset "
    "&nbsp;·&nbsp; 2 A discharge, 24 °C</div>"
    "<div class='masthead'>Cell health, read from a partial discharge</div>"
    "<div class='standfirst'>Capacity normally needs a full controlled cycle to "
    "measure. This reads state of health and remaining cycles from the segment where "
    "terminal voltage falls from 4.0 V to 3.6 V — a window that occurs in ordinary "
    "use.</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='hairline'>", unsafe_allow_html=True)

mode = st.radio("Input", ["Read a cell file", "Set features by hand"],
                horizontal=True, label_visibility="collapsed")

# ----------------------------------------------------------------------
# Mode 1 — cell file
# ----------------------------------------------------------------------
if mode == "Read a cell file":
    up = st.file_uploader("NASA PCoE cell file (e.g. B0005.mat)", type=["mat"])

    if up is None:
        st.markdown(
            "<div class='noteblock'>Load a cell file to chart its full degradation "
            "history. Files come from the NASA Prognostics Center of Excellence "
            "battery ageing dataset — B0005, B0006, B0007 and B0018.</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    mat = sio.loadmat(up)
    key = [k for k in mat if not k.startswith("__")][0]
    cycles = mat[key]["cycle"][0, 0][0]
    dis = [c for c in cycles if c["type"][0] == "discharge"]

    rows = [window_features(c) for c in dis]
    kept = [r for r in rows if r is not None]

    if not kept:
        st.error("No usable discharge cycles in this file. A cycle must cross from "
                 "4.0 V down to 3.6 V for the features to be extracted.")
        st.stop()

    df = pd.DataFrame(kept)
    df["cycle"] = range(1, len(df) + 1)
    df["soh"] = model.predict(df[FEATURES])

    df = add_rolling_features(df)
    ready = df[RUL_FEATURES].notna().all(axis=1)
    if ready.any():
        df.loc[ready, "rul"] = np.maximum(
            rul_model.predict(df.loc[ready, RUL_FEATURES]), 0.0)

    latest = df["soh"].iloc[-1]
    label, colour = status_of(latest)

    rul_cell = (spec("Remaining life", f"{df.loc[ready, 'rul'].iloc[-1]:.0f}", "cyc",
                     f"± {RUL_RMSE:.1f} cycles, cross-cell")
                if ready.any()
                else spec("Remaining life", "—", "",
                          f"needs {MIN_CYCLES_FOR_RUL}+ cycles"))

    st.markdown(
        "<div class='specstrip'>"
        + spec("Cell", key)
        + spec("Cycles read", f"{len(df)}")
        + spec("State of health", f"{latest:.1f}", "%", "latest cycle")
        + rul_cell
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(health_gauge(latest, label, colour), unsafe_allow_html=True)

    if latest < 75:
        st.warning(
            "This cell is at or past its end-of-life threshold. The remaining-life "
            "model was trained only up to end of life, so beyond that point it is "
            "extrapolating and the figure should not be relied on."
        )

    st.markdown("<hr class='hairline'>", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})

    axes[0].plot(df["cycle"], df["soh"], lw=1.8, color=TRACE)
    axes[0].axhline(80, color=CAUTION, ls=(0, (4, 3)), lw=0.9)
    axes[0].axhline(70, color=CRITICAL, ls=(0, (4, 3)), lw=0.9)
    axes[0].annotate("80% ageing", xy=(0.995, 80),
                     xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7.5, color=CAUTION)
    axes[0].annotate("70% end of life", xy=(0.995, 70),
                     xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7.5, color=CRITICAL)
    axes[0].set_ylabel("State of health  (%)")
    style_axes(axes[0])

    if ready.any():
        sub = df[ready]
        axes[1].fill_between(sub["cycle"],
                             np.maximum(sub["rul"] - RUL_RMSE, 0),
                             sub["rul"] + RUL_RMSE,
                             color=CAUTION, alpha=0.13, lw=0)
        axes[1].plot(sub["cycle"], sub["rul"], lw=1.8, color=CAUTION)
        axes[1].annotate(f"shaded band ± {RUL_RMSE:.1f} cycles",
                         xy=(0.995, 0.88), xycoords="axes fraction",
                         ha="right", fontsize=7.5, color=MUTED)
    else:
        axes[1].text(0.5, 0.5, f"needs {MIN_CYCLES_FOR_RUL}+ cycles",
                     ha="center", va="center", transform=axes[1].transAxes,
                     color=MUTED, fontsize=9)

    axes[1].set_ylabel("Remaining life  (cycles)")
    axes[1].set_xlabel("Discharge cycle")
    style_axes(axes[1])

    fig.tight_layout()
    st.pyplot(fig, transparent=True)

    with st.expander("Per-cycle readings"):
        cols = ["cycle"] + FEATURES + ["soh"]
        cols += [c for c in RUL_FEATURES if c not in cols]
        if "rul" in df.columns:
            cols += ["rul"]
        st.dataframe(df[cols].round(3), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# Mode 2 — manual features
# ----------------------------------------------------------------------
else:
    st.markdown(
        "<div class='noteblock'>Set the three measured features directly to see how "
        "each one moves the health estimate.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    win_time = c1.slider("Window traverse time (s)", 600.0, 1500.0, 1200.0, 10.0,
                         help="Seconds taken to fall from 4.0 V to 3.6 V. "
                              "A worn cell crosses it faster.")
    ir_proxy = c2.slider("Voltage drop at load (V)", 0.10, 0.50, 0.22, 0.01,
                         help="Immediate drop when the load is applied. "
                              "Rises as internal resistance grows.")
    win_shape = c3.slider("Window shape ratio", 0.15, 0.50, 0.29, 0.01,
                          help="Fraction of the window spent above 3.8 V.")

    x = pd.DataFrame([{
        "win_time": win_time,
        "ir_proxy": ir_proxy,
        "win_shape": win_shape,
    }])[FEATURES]

    soh = float(model.predict(x)[0])
    label, colour = status_of(soh)

    st.markdown(
        "<div class='specstrip'>"
        + spec("State of health", f"{soh:.1f}", "%", "from the features above")
        + spec("Remaining life", "—", "", "needs consecutive cycles")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(health_gauge(soh, label, colour), unsafe_allow_html=True)

    st.markdown(
        "<div class='noteblock' style='margin-top:0.9rem'>Remaining life is not "
        "available here. It reads how these quantities <b>trend</b> across "
        "consecutive cycles, so it needs a cell file rather than one set of "
        "values.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------
st.markdown("<hr class='hairline'>", unsafe_allow_html=True)

n1, n2, n3 = st.columns(3)

with n1:
    st.markdown(
        "<div class='eyebrow'>Method</div>"
        "<div class='noteblock'>Three features come from the segment where terminal "
        "voltage falls from 4.0 V to 3.6 V: the time to cross it, the immediate drop "
        "when the load is applied (a proxy for internal resistance), and the shape of "
        "the curve inside the window. Remaining life uses a second model built on how "
        "those quantities trend over recent cycles rather than their value at any one "
        "cycle.</div>",
        unsafe_allow_html=True,
    )

with n2:
    st.markdown(
        "<div class='eyebrow'>Validation</div>"
        "<div class='noteblock'>Both models are tested leave-one-cell-out: trained on "
        "every cell but one, then scored on the cell they have never seen. "
        "<b>State of health — 1.69 percentage points RMSE</b> across four cells. "
        "<b>Remaining life — 10.7 cycles RMSE</b> across three cells, against "
        "lifetimes of 97 to 125 cycles.</div>",
        unsafe_allow_html=True,
    )

with n3:
    st.markdown(
        "<div class='eyebrow'>Limits</div>"
        "<div class='noteblock'>Trained on 18650 cells of one chemistry at 24 °C under "
        "a fixed 2 A protocol; other chemistries, temperatures and duty cycles are "
        "untested. The remaining-life model is built from three cells, and in "
        "leave-one-cell-out testing it tended to <b>over-estimate</b> remaining life "
        "late in a cell's life — the optimistic, less safe direction. An offline "
        "analysis tool, not a battery management system.</div>",
        unsafe_allow_html=True,
    )

