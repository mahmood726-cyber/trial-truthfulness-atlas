"""Flag 2 — relative N drift between registered enrolment and MA-pooled N."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from tta import config

# Negation phrase + the number that immediately follows it. Replacing the WHOLE
# match (including the count) is what prevents the Verquvo regression: matching
# only the phrase leaves the number behind for NUMBER_RE to pick up if it
# happens to appear before any legitimate count.
NEGATION_BEFORE = re.compile(
    r"(?:not|non[- ]?|never|no(?:t)?[- ]?(?:yet)?|withdrawn[- ]?before)"
    r"\s*[-]?\s*(?:randomi[sz]ed|enrolled|analy[sz]ed)"
    r"\s+(?:\d{1,3}(?:[,\s]\d{3})+|\d+)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b(\d{1,3}(?:[, ]\d{3})*|\d+)\b")


def extract_first_n(text: str) -> Optional[int]:
    cleaned = NEGATION_BEFORE.sub(" __SCRUBBED__ ", text)
    for m in NUMBER_RE.finditer(cleaned):
        token = m.group(1).replace(",", "").replace(" ", "")
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def classify(
    registered_n: Optional[int],
    ma_n: Optional[int],
    threshold: Optional[float] = None,
) -> str:
    threshold = threshold if threshold is not None else config.N_DRIFT_THRESHOLD
    if registered_n is None or ma_n is None:
        return "unscoreable"
    try:
        registered_n = int(registered_n)
        ma_n = int(ma_n)
    except (TypeError, ValueError):
        return "unscoreable"
    if registered_n <= 0:
        return "unscoreable"
    rel = abs(ma_n - registered_n) / registered_n
    # Strict `>`, not `>=`. Boundary case (drift exactly at the threshold) is
    # treated as not_flagged. Matches FDAAA audit convention ("more than X%").
    return "flagged" if rel > threshold else "not_flagged"


def compute_dataframe(df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    out["n_drift"] = [
        classify(row.get("registered_n"), row.get("ma_n"), threshold=threshold)
        for _, row in df.iterrows()
    ]
    return out
