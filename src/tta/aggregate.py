"""MA-level rollup of the atlas.

Effect-scale convention: `ma_effect_log`, `ma_ci_low`, `ma_ci_high`
columns are assumed to be on the LOG scale (logHR / logOR / logRR / SMD),
matching the Pairwise70 convention. The `crosses_null` predicate tests
whether 0.0 lies inside the pooled CI bracket. If a future caller passes
natural-scale CIs, the null is 1.0 and `crosses_null` would be wrong —
log-transform first.

v0.2.0: pooled effect + CI come from a real random-effects meta-analysis
(DL τ² + HKSJ-adjusted variance per `tta.meta.random_effects_pool`),
not from the first trial's CI as in v0.1.x.
"""

from __future__ import annotations

import pandas as pd

from tta import meta

ANY_FLAG_PREDICATES = {
    "outcome_drift": {"substantively_different"},
    "n_drift": {"flagged"},
    "direction_concordance": {"flipped"},
    "results_posting": {"required_not_posted"},
}

_ROLLUP_COLUMNS = [
    "review_id", "review_doi", "n_trials", "n_unbridgeable",
    "n_trials_with_any_flag", "n_outcome_drift_substantive",
    "n_n_drift_flagged", "n_direction_flipped",
    "n_results_required_not_posted",
    "pooled_effect_log", "pooled_ci_low", "pooled_ci_high",
    "tau2", "pool_method", "crosses_null",
]


def _trial_has_any_flag(row: pd.Series) -> bool:
    for col, hits in ANY_FLAG_PREDICATES.items():
        if row.get(col) in hits:
            return True
    return False


def _pool_group(group: pd.DataFrame) -> meta.PoolResult:
    """Pool the trials in one MA group using random-effects DL+HKSJ.

    Inputs come from `ma_effect_log`, `ma_ci_low`, `ma_ci_high`.
    Variance is recovered from CI bounds via `meta.variance_from_ci`.
    Trials with missing effect or unrecoverable variance are dropped.
    """
    if "ma_effect_log" not in group.columns:
        return meta.random_effects_pool([], [])
    effects: list[float] = []
    variances: list[float] = []
    for _, row in group.iterrows():
        eff = row.get("ma_effect_log")
        lo = row.get("ma_ci_low")
        hi = row.get("ma_ci_high")
        if pd.isna(eff) or pd.isna(lo) or pd.isna(hi):
            continue
        v = meta.variance_from_ci(float(eff), float(lo), float(hi))
        if v is None or v <= 0:
            continue
        effects.append(float(eff))
        variances.append(v)
    return meta.random_effects_pool(effects, variances)


def ma_rollup(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame(columns=_ROLLUP_COLUMNS)

    rows = []
    for review_id, group in atlas.groupby("review_id", sort=True):
        any_flag = group.apply(_trial_has_any_flag, axis=1)
        pool = _pool_group(group)
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
            "pooled_effect_log": pool.mu,
            "pooled_ci_low": pool.ci_low,
            "pooled_ci_high": pool.ci_high,
            "tau2": pool.tau2,
            "pool_method": pool.method,
            "crosses_null": pool.crosses_null,
        })
    return pd.DataFrame(rows)
