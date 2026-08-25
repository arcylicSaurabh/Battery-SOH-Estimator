"""
Battery State of Health and Remaining Useful Life estimator — Streamlit app.

Upload a NASA PCoE .mat cell file, or move the sliders manually.
SOH:  3-feature linear regression trained on B0005/B0006/B0007/B0018.
RUL:  3-feature random forest trained on B0005/B0006/B0018.
"""

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st
import matplotlib.pyplot as plt

from features import (
    window_features,
    add_rolling_features,
    MIN_CYCLES_FOR_RUL,
)


def status_of(soh):
    if soh >= 90:
        return "Healthy", "normal"
    if soh >= 80:
        return "Ageing", "off"
    return "Replace soon", "inverse"


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Battery SOH Estimator", layout="wide")

st.title("Li-ion Battery Health and Remaining Life Estimator")
st.caption(
    "Estimates battery health and remaining cycles from a partial discharge "
    "segment (4.0 V to 3.6 V), without requiring a full capacity test."
)


@st.cache_resource
def load_models():
    return joblib.load("soh_model.pkl"), joblib.load("rul_model.pkl")


soh_bundle, rul_bundle = load_models()
model, FEATURES = soh_bundle["model"], soh_bundle["features"]
rul_model, RUL_FEATURES = rul_bundle["model"], rul_bundle["features"]
RUL_RMSE = rul_bundle.get("cv_rmse_cycles", 10.7)

mode = st.radio("Input", ["Upload a .mat cell file", "Enter features manually"],
                horizontal=True)

# ----------------------------------------------------------------------
# Mode 1 — upload a cell file
# ----------------------------------------------------------------------
if mode == "Upload a .mat cell file":
    up = st.file_uploader("NASA PCoE cell file (e.g. B0005.mat)", type=["mat"])

    if up is None:
        st.info(
            "Upload a cell file to see its full degradation curve. "
            "Files come from the NASA Prognostics Center of Excellence "
            "battery aging dataset."
        )
        st.stop()

    mat = sio.loadmat(up)
    key = [k for k in mat if not k.startswith("__")][0]
    cycles = mat[key]["cycle"][0, 0][0]
    dis = [c for c in cycles if c["type"][0] == "discharge"]

    rows = [window_features(c) for c in dis]
    kept = [r for r in rows if r is not None]

    if not kept:
        st.error("No usable discharge cycles found in this file.")
        st.stop()

    df = pd.DataFrame(kept)
    df["cycle"] = range(1, len(df) + 1)
    df["soh"] = model.predict(df[FEATURES])

    # Rolling features for RUL. Early cycles stay NaN until the windows fill.
    df = add_rolling_features(df)
    ready = df[RUL_FEATURES].notna().all(axis=1)
    if ready.any():
        df.loc[ready, "rul"] = np.maximum(
            rul_model.predict(df.loc[ready, RUL_FEATURES]), 0.0)

    latest = df["soh"].iloc[-1]
    label, style = status_of(latest)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cell", key)
    c2.metric("Cycles analysed", len(df))
    c3.metric("Latest predicted SOH", f"{latest:.1f} %", label, delta_color=style)

    if ready.any():
        latest_rul = df.loc[ready, "rul"].iloc[-1]
        c4.metric("Predicted RUL", f"{latest_rul:.0f} cycles",
                  f"±{RUL_RMSE:.1f}", delta_color="off")
    else:
        c4.metric("Predicted RUL", "—")
        st.info(
            f"RUL needs at least {MIN_CYCLES_FOR_RUL} cycles of history to "
            f"build its rolling trend features — this file has {len(df)}."
        )

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(df["cycle"], df["soh"], lw=1.5)
    axes[0].axhline(80, color="orange", ls="--", lw=1, label="80% — ageing")
    axes[0].axhline(70, color="red", ls="--", lw=1, label="70% — end of life")
    axes[0].set_ylabel("Predicted SOH (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if ready.any():
        sub = df[ready]
        axes[1].plot(sub["cycle"], sub["rul"], lw=1.5, color="tab:green")
        axes[1].fill_between(sub["cycle"],
                             np.maximum(sub["rul"] - RUL_RMSE, 0),
                             sub["rul"] + RUL_RMSE,
                             color="tab:green", alpha=0.15,
                             label=f"±{RUL_RMSE:.1f} cycles")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Not enough cycles for RUL",
                     ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_ylabel("Predicted RUL (cycles)")
    axes[1].set_xlabel("Discharge cycle")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Extracted features"):
        cols = ["cycle"] + FEATURES + ["soh"]
        cols += [c for c in RUL_FEATURES if c not in cols]
        if "rul" in df.columns:
            cols += ["rul"]
        st.dataframe(df[cols].round(3), use_container_width=True)

# ----------------------------------------------------------------------
# Mode 2 — manual sliders
# ----------------------------------------------------------------------
else:
    st.write("Move the sliders to see how each feature affects the estimate.")

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
    label, style = status_of(soh)

    st.metric("Predicted State of Health", f"{soh:.1f} %", label,
              delta_color=style)
    st.progress(min(max(soh / 100, 0.0), 1.0))

    st.caption(
        "RUL is not available in this mode. It depends on rolling trend "
        "features computed across consecutive cycles, so it needs a cell "
        "file rather than a single set of values."
    )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.divider()

st.markdown(
    """
**How it works** — Battery capacity normally requires a full controlled
charge-discharge cycle to measure, which is impractical on a vehicle in service.
This model estimates it instead from a short segment of the discharge curve that
occurs during ordinary use.

Three features are taken from the segment where terminal voltage falls from
4.0 V to 3.6 V: the time taken to cross it, the immediate voltage drop when the
load is applied (a proxy for internal resistance), and the shape of the curve
within the window. Remaining useful life uses a second model built on how those
quantities *trend* over recent cycles rather than their values at one cycle.

**Accuracy** — Both models are validated by leave-one-cell-out testing, where
the model is trained on all cells but one and tested on the cell it has never
seen. SOH: mean RMSE 1.69 percentage points across four cells. RUL: mean RMSE
10.7 cycles across three cells, against cell lifetimes of 97 to 125 cycles.

**Limits** — Trained on 18650 cells of one chemistry, cycled at 24 °C under a
fixed 2 A discharge protocol. Accuracy on other chemistries, temperatures or
duty cycles is untested. The RUL model is built from three cells only, and in
leave-one-cell-out testing it tended to over-estimate remaining life late in a
cell's life — the optimistic and therefore less safe direction. Treat the
figure as an indicative trend, not a maintenance guarantee. This is an offline
analysis tool, not a replacement for a battery management system.
"""
)

