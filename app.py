"""
Battery State of Health estimator — Streamlit app.

Upload a NASA PCoE .mat cell file, or move the sliders manually.
Model: 3-feature linear regression trained on B0005/B0006/B0007/B0018.
"""

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Feature extraction — MUST stay identical to the notebook version
# ----------------------------------------------------------------------
V_HI, V_MID, V_LO = 4.0, 3.8, 3.6


def window_features(cyc):
    d = cyc['data'][0, 0]
    t = d['Time'][0].astype(float)
    v = d['Voltage_measured'][0].astype(float)
    tp = d['Temperature_measured'][0].astype(float)

    end = int(np.argmin(v))
    t, v, tp = t[:end + 1], v[:end + 1], tp[:end + 1]

    def first_below(th):
        idx = np.where(v <= th)[0]
        return idx[0] if len(idx) else None

    a, m, b = first_below(V_HI), first_below(V_MID), first_below(V_LO)
    if a is None or m is None or b is None or b <= a:
        return None

    tw = t[a:b + 1]
    win = tw[-1] - tw[0]

    return {
        'win_time':  win,
        'ir_proxy':  v[0] - v[2],
        'win_shape': (t[m] - t[a]) / win,
    }


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

st.title("Li-ion Battery State of Health Estimator")
st.caption(
    "Estimates battery health from a partial discharge segment (4.0 V to 3.6 V), "
    "without requiring a full capacity test."
)

bundle = joblib.load("soh_model.pkl")
model, FEATURES = bundle["model"], bundle["features"]

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

    latest = df["soh"].iloc[-1]
    label, style = status_of(latest)

    c1, c2, c3 = st.columns(3)
    c1.metric("Cell", key)
    c2.metric("Cycles analysed", len(df))
    c3.metric("Latest predicted SOH", f"{latest:.1f} %", label, delta_color=style)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["cycle"], df["soh"], lw=1.5)
    ax.axhline(80, color="orange", ls="--", lw=1, label="80% — ageing")
    ax.axhline(70, color="red", ls="--", lw=1, label="70% — end of life")
    ax.set_xlabel("Discharge cycle")
    ax.set_ylabel("Predicted SOH (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)

    with st.expander("Extracted features"):
        st.dataframe(df[["cycle"] + FEATURES + ["soh"]].round(3),
                     use_container_width=True)

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
within the window.

**Accuracy** — Validated by leave-one-cell-out testing on four cells, where the
model is trained on three cells and tested on a fourth it has never seen.
Mean RMSE 1.69 percentage points of SOH.

**Limits** — Trained on four 18650 cells of one chemistry, cycled at 24 °C under
a fixed 2 A discharge protocol. Accuracy on other chemistries, temperatures or
duty cycles is untested. This is an offline analysis tool, not a replacement for
a battery management system.
"""
)
