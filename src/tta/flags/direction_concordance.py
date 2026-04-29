"""Flag 3 — direction concordance between registered and MA-pooled effects.

Both inputs assumed on the LOG scale (logHR / logOR / logRR / SMD). AACT
HR/OR/RR values arrive on natural scale and must be log-transformed via
`hr_to_log_effect` before classification.

v0.1.0 scope: pipeline only joins AACT `outcome_analyses` rows where
`param_type == "Hazard Ratio"`. Risk Ratio, Odds Ratio, and Mean Difference
joins are deferred to v0.2.0 (RR/OR also need log; MD does not).
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from tta import config


def hr_to_log_effect(hr: float) -> float:
    return math.log(hr)


def classify(
    registered_effect: Optional[float],
    ma_effect: Optional[float],
    epsilon: Optional[float] = None,
) -> str:
    eps = epsilon if epsilon is not None else config.DIRECTION_EPSILON
    # Handle None and NaN (pandas uses NaN for missing values)
    if registered_effect is None or ma_effect is None:
        return "unscoreable"
    if pd.isna(registered_effect) or pd.isna(ma_effect):
        return "unscoreable"
    if abs(registered_effect) < eps or abs(ma_effect) < eps:
        return "unscoreable"
    return "concordant" if (registered_effect * ma_effect) > 0 else "flipped"


def compute_dataframe(df: pd.DataFrame, epsilon: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    out["direction_concordance"] = [
        classify(row["registered_effect_log"], row["ma_effect_log"], epsilon=epsilon)
        for _, row in df.iterrows()
    ]
    return out
