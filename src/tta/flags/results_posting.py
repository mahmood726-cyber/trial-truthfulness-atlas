"""Flag 4 — FDAAA results-posting compliance."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd

FDAAA_INTERVENTION_TYPES = frozenset({"drug", "biological", "device"})
FDAAA_DEADLINE_DAYS = 365  # 12 months between completion and required posting


def _is_fdaaa_applicable(
    study_type: Optional[str],
    intervention_types: Iterable[str],
) -> bool:
    if (study_type or "").lower() != "interventional":
        return False
    types = {(t or "").lower() for t in intervention_types}
    return bool(types & FDAAA_INTERVENTION_TYPES)


def classify(
    results_first_posted_date: Optional[date],
    completion_date: Optional[date],
    study_type: Optional[str],
    intervention_types: Iterable[str],
    snapshot_date: date,
) -> str:
    if not _is_fdaaa_applicable(study_type, intervention_types):
        return "not_required"
    if completion_date is None:
        return "unscoreable"
    if completion_date > snapshot_date - timedelta(days=FDAAA_DEADLINE_DAYS):
        return "unscoreable"
    if results_first_posted_date is not None:
        return "posted"
    return "required_not_posted"


def compute_dataframe(df: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    out = df.copy()
    out["results_posting"] = [
        classify(
            results_first_posted_date=row.get("results_first_posted_date"),
            completion_date=row.get("completion_date"),
            study_type=row.get("study_type"),
            intervention_types=row.get("intervention_types") or [],
            snapshot_date=snapshot_date,
        )
        for _, row in df.iterrows()
    ]
    return out
