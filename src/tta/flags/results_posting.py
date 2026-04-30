"""Flag 4 — FDAAA results-posting compliance."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd

# `device` is included for v0.1.0 simplicity, but FDAAA 801 / 42 CFR
# 11.22(b)(1) only covers Applicable Device Clinical Trials (significant-risk
# IDE trials). Non-SR device trials are exempt. v0.2.0 should join
# AACT `expanded_access_info` / `study_design_info` to approximate IDE
# status; until then, v0.1.0 will overcount on device trials.
FDAAA_INTERVENTION_TYPES = frozenset({"drug", "biological", "device"})

# 12 months in 42 CFR 11.64(b)(1)(ii); we use 365 days as a calendar
# approximation (off-by-one across leap years). The anchor is the trial's
# primary_completion_date; pipeline._enrich_with_aact resolves that and
# falls back to completion_date only when primary is missing.
FDAAA_DEADLINE_DAYS = 365


def _is_fdaaa_applicable(
    study_type: Optional[str],
    intervention_types: Iterable[str],
) -> bool:
    if not isinstance(study_type, str) or study_type.lower() != "interventional":
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
