"""MA-level rollup of the atlas."""

from __future__ import annotations

import pandas as pd

ANY_FLAG_PREDICATES = {
    "outcome_drift": {"substantively_different"},
    "n_drift": {"flagged"},
    "direction_concordance": {"flipped"},
    "results_posting": {"required_not_posted"},
}


def _trial_has_any_flag(row: pd.Series) -> bool:
    for col, hits in ANY_FLAG_PREDICATES.items():
        if row.get(col) in hits:
            return True
    return False


def _crosses_null_from_first_ci(group: pd.DataFrame) -> bool | None:
    if "ma_ci_low" not in group.columns or "ma_ci_high" not in group.columns:
        return None
    valid = group.dropna(subset=["ma_ci_low", "ma_ci_high"])
    if valid.empty:
        return None
    lo = float(valid["ma_ci_low"].iloc[0])
    hi = float(valid["ma_ci_high"].iloc[0])
    return lo <= 0.0 <= hi


def ma_rollup(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame(columns=[
            "review_id", "review_doi", "n_trials", "n_unbridgeable",
            "n_trials_with_any_flag", "n_outcome_drift_substantive",
            "n_n_drift_flagged", "n_direction_flipped",
            "n_results_required_not_posted", "crosses_null",
        ])

    rows = []
    for review_id, group in atlas.groupby("review_id", sort=True):
        any_flag = group.apply(_trial_has_any_flag, axis=1)
        rows.append({
            "review_id": review_id,
            "review_doi": group["review_doi"].iloc[0] if "review_doi" in group else None,
            "n_trials": len(group),
            "n_unbridgeable": int((group["bridge_method"] == "unbridgeable").sum()),
            "n_trials_with_any_flag": int(any_flag.sum()),
            "n_outcome_drift_substantive": int((group["outcome_drift"] == "substantively_different").sum()),
            "n_n_drift_flagged": int((group["n_drift"] == "flagged").sum()),
            "n_direction_flipped": int((group["direction_concordance"] == "flipped").sum()),
            "n_results_required_not_posted": int((group["results_posting"] == "required_not_posted").sum()),
            "crosses_null": _crosses_null_from_first_ci(group),
        })
    return pd.DataFrame(rows)
