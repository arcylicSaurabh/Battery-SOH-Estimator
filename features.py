"""
Feature extraction for the battery SOH / RUL project.

This module is the single source of truth for feature engineering.
Import it in BOTH the training notebook and the Streamlit app so the two
can never drift apart:

    from features import window_features, add_rolling_features

Any change here must be followed by retraining, since the saved models
depend on these exact definitions.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
V_HI, V_MID, V_LO = 4.0, 3.8, 3.6

ROLL_WINDOW = 20
MIN_PERIODS = 5

SOH_FEATURES = ['win_time', 'ir_proxy', 'win_shape']
RUL_FEATURES = ['wt_mean', 'win_time', 'ir_slope']

# ir_slope needs a .diff() first, which consumes one row on top of
# MIN_PERIODS. This is the real minimum cycle count for an RUL estimate.
MIN_CYCLES_FOR_RUL = MIN_PERIODS + 1


# ----------------------------------------------------------------------
# Per-cycle features (partial discharge window, 4.0 V -> 3.6 V)
# ----------------------------------------------------------------------
def window_features(cyc):
    """Extract the three SOH features from one discharge cycle.

    Returns None if the cycle never crosses the full voltage window,
    which happens on malformed or truncated cycles.
    """
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


# ----------------------------------------------------------------------
# Rolling trend features (needed by the RUL model)
# ----------------------------------------------------------------------
def add_rolling_features(d, w=ROLL_WINDOW, min_periods=MIN_PERIODS):
    """Add rolling trend features to a per-cycle feature frame.

    Rows are expected in cycle order. Early rows carry NaN until the
    rolling windows fill; callers decide whether to drop them.
    """
    d = d.copy()
    d['wt_mean'] = d['win_time'].rolling(w, min_periods=min_periods).mean()
    d['wt_slope'] = d['win_time'].diff().rolling(w, min_periods=min_periods).mean()
    d['wt_drop'] = d['win_time'] / d['win_time'].iloc[0]
    d['ir_slope'] = d['ir_proxy'].diff().rolling(w, min_periods=min_periods).mean()
    return d


def add_rul_features(d, eol, w=ROLL_WINDOW, min_periods=MIN_PERIODS):
    """Training-time helper: truncate at end of life, label RUL, drop NaN rows.

    Used by the notebook. The app calls add_rolling_features directly,
    since it has no EOL label for an unseen cell.
    """
    d = d[d['cycle'] <= eol].copy()
    d['rul'] = eol - d['cycle']
    d = add_rolling_features(d, w=w, min_periods=min_periods)
    return d.dropna()
