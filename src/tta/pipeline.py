"""End-to-end pipeline runner for the v0.1.0 fixture and (later) full sweep."""

from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from tta import aggregate, bridge, cardio_filter, ingest
from tta.flags import (
    direction_concordance,
    n_drift,
    outcome_drift,
    results_posting,
)
from tta.judge import cache as cache_mod

logger = logging.getLogger(__name__)


def _select_primary_hr(
    aact_outcome_analyses: pd.DataFrame,
    outcomes: Optional[pd.DataFrame],
) -> pd.Series:
    """Return a Series keyed by nct_id with the HR param_value to use.

    When the AACT `outcomes` table is available, join outcome_analyses to
    outcomes on outcome_id and pick the row whose outcome_type == 'Primary'.
    If a trial has no Primary-type HR row, fall back to its first available
    HR row (by row order).  Without the outcomes table, fall back to the
    old drop_duplicates("nct_id") behaviour.
    """
    hr_rows = aact_outcome_analyses[
        aact_outcome_analyses["param_type"] == "Hazard Ratio"
    ].copy()
    if hr_rows.empty:
        return pd.Series(dtype=float)

    if outcomes is not None and "outcome_id" in hr_rows.columns and "id" in outcomes.columns:
        # Join to classify each HR row as Primary / non-Primary.
        merged = hr_rows.merge(
            outcomes[["id", "outcome_type"]].rename(columns={"id": "outcome_id"}),
            on="outcome_id",
            how="left",
        )
        # Normalise case so 'Primary' and 'primary' both match.
        merged["_is_primary"] = merged["outcome_type"].str.lower().str.strip() == "primary"

        # Per-trial: prefer primary rows; fall back to first available row.
        selected = []
        for nct, group in merged.groupby("nct_id"):
            primary = group[group["_is_primary"]]
            chosen = primary.iloc[0] if not primary.empty else group.iloc[0]
            selected.append(chosen)
        if not selected:
            return pd.Series(dtype=float)
        best = pd.DataFrame(selected)
    else:
        # Legacy path: no outcomes table supplied; keep first HR row per trial.
        best = hr_rows.drop_duplicates("nct_id")

    return best.set_index("nct_id")["param_value"].astype(float)


def _enrich_with_aact(
    bridged: pd.DataFrame,
    aact_studies: pd.DataFrame,
    aact_design_outcomes: pd.DataFrame,
    aact_calculated: pd.DataFrame,
    aact_outcome_analyses: pd.DataFrame,
    aact_interventions: pd.DataFrame,
    outcomes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = bridged.copy()

    primary_outcomes = (
        aact_design_outcomes[aact_design_outcomes["outcome_type"] == "primary"]
        .drop_duplicates("nct_id")
        .set_index("nct_id")["measure"]
    )
    out["registered_outcome"] = out["nct_id"].map(primary_outcomes)

    enrolment = (
        aact_calculated.drop_duplicates("nct_id")
        .set_index("nct_id")["actual_enrollment"]
        .astype("Int64")
    )
    out["registered_n"] = out["nct_id"].map(enrolment)

    hr = _select_primary_hr(aact_outcome_analyses, outcomes)
    # math.log raises ValueError on hr <= 0; AACT param_value is free-text and
    # may legitimately or accidentally hold non-positive values. Guard the call.
    out["registered_effect_log"] = out["nct_id"].map(
        lambda nct: math.log(hr[nct])
        if nct in hr.index and not pd.isna(hr[nct]) and hr[nct] > 0
        else None
    )

    studies = aact_studies.drop_duplicates("nct_id").set_index("nct_id")
    # FDAAA 42 CFR 11.64(b)(1)(ii) anchors the 12-month posting deadline to
    # primary_completion_date; completion_date is the LAST data-collection date
    # (primary + secondary endpoints) and is always >= primary_completion_date.
    # Using completion_date would give trials more clock time than statute
    # allows, undercounting violations. Fall back to completion_date only when
    # primary_completion_date is missing.
    if "primary_completion_date" in studies.columns:
        primary_cd = out["nct_id"].map(studies["primary_completion_date"])
        completion = out["nct_id"].map(studies["completion_date"])
        out["completion_date"] = primary_cd.where(primary_cd.notna(), completion)
    else:
        out["completion_date"] = out["nct_id"].map(studies["completion_date"])
    out["results_first_posted_date"] = out["nct_id"].map(studies["results_first_posted_date"])
    out["study_type"] = out["nct_id"].map(studies["study_type"])

    interventions_grouped = (
        aact_interventions.groupby("nct_id")["intervention_type"]
        .apply(list)
    )
    out["intervention_types"] = out["nct_id"].map(interventions_grouped)
    out["intervention_types"] = out["intervention_types"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    return out


def _atlas_columns_in_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return a NEW DataFrame with the atlas's pinned 16-column schema.
    Missing columns are filled with NaN. Does NOT mutate the input."""
    cols = [
        "nct_id", "review_id", "review_doi", "Study",
        "bridge_method", "bridge_confidence",
        "registered_outcome", "ma_extracted_outcome", "outcome_drift",
        "registered_n", "ma_n", "n_drift",
        "registered_effect_log", "ma_effect_log", "direction_concordance",
        "results_posting",
    ]
    return df.reindex(columns=cols)


def run_5trial_fixture(
    fixtures_dir: Path,
    out_dir: Path,
    snapshot_date: date,
    ollama_client,
    pairwise70_dir_override: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aact_dir = fixtures_dir / "aact_sample"
    pw_dir = pairwise70_dir_override or (fixtures_dir / "pairwise70_sample")

    logger.info("loading Pairwise70 from %s", pw_dir)
    pw = ingest.load_pairwise70_dir(pw_dir)
    pw = cardio_filter.filter_pairwise70(pw)
    pw["ma_n"] = pw.get("Experimental.N", 0).fillna(0).astype(int) + pw.get("Control.N", 0).fillna(0).astype(int)
    pw["ma_extracted_outcome"] = pw.get("Analysis.name")
    pw["ma_effect_log"] = pw.get("Mean")
    pw["ma_ci_low"] = pw.get("CI.start")
    pw["ma_ci_high"] = pw.get("CI.end")
    logger.info("filtered to %d cardio trials across %d reviews",
                len(pw), pw["review_id"].nunique() if not pw.empty else 0)

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    # Load tables needed for bridge methods 2 and 4 before the bridge call.
    studies = ingest.load_aact_table(aact_dir, "studies")
    id_info_path = aact_dir / "id_information.txt"
    aact_id_information = ingest.load_aact_table(aact_dir, "id_information") if id_info_path.is_file() else None
    bridged = bridge.bridge_pairwise70(
        pw, dossiergap=dg,
        aact_id_information=aact_id_information,
        aact_studies=studies,
    )
    logger.info("Flag 0 bridge resolution: %d/%d (%.1f%%) of trials bridged to NCT",
                int((bridged["bridge_method"] != "unbridgeable").sum()), len(bridged),
                100 * bridge.resolution_rate(bridged))
    design_outcomes = ingest.load_aact_table(aact_dir, "design_outcomes")
    calculated = ingest.load_aact_table(aact_dir, "calculated_values")
    outcome_analyses = ingest.load_aact_table(aact_dir, "outcome_analyses")
    interventions = ingest.load_aact_table(aact_dir, "interventions")
    # outcomes table maps outcome_id -> outcome_type so _enrich_with_aact can
    # prefer the Primary-outcome HR row when a trial has multiple HR rows.
    aact_outcomes_path = aact_dir / "outcomes.txt"
    aact_outcomes = ingest.load_aact_table(aact_dir, "outcomes") if aact_outcomes_path.is_file() else None

    enriched = _enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions, outcomes=aact_outcomes,
    )

    # Convert NaN in string/object columns to None so downstream guards fire correctly.
    # pandas NaN is truthy; Python None is falsy — flag compute_one guards use `not x`.
    # Note: in pandas 3.x, StringDtype columns ignore `where(other=None)` because pandas
    # re-encodes None as NaN for that dtype. Downstream consumers guard with pd.isnull()
    # explicitly rather than relying solely on None-ness.
    for col in ("registered_outcome", "ma_extracted_outcome", "study_type"):
        if col in enriched.columns:
            enriched[col] = enriched[col].where(pd.notna(enriched[col]), other=None)

    judge_cache = cache_mod.JudgeCache(out_dir / "judge_cache")
    cache_size_before = len(judge_cache)
    enriched = outcome_drift.compute_dataframe(enriched, client=ollama_client, cache=judge_cache)
    cache_size_after = len(judge_cache)
    logger.info("Flag 1 outcome-drift: %d new cache entries (%d total)",
                cache_size_after - cache_size_before, cache_size_after)
    enriched = n_drift.compute_dataframe(enriched)
    enriched = direction_concordance.compute_dataframe(enriched)
    # Convert to Python date; replace NaT with None so results_posting.classify
    # correctly treats missing posted-date as not-yet-posted (NaT is not None in Python).
    def _to_date_or_none(series: pd.Series) -> pd.Series:
        converted = pd.to_datetime(series, errors="coerce").dt.date
        return converted.where(converted.notna(), other=None)

    enriched["completion_date"] = _to_date_or_none(enriched["completion_date"])
    enriched["results_first_posted_date"] = _to_date_or_none(enriched["results_first_posted_date"])
    enriched = results_posting.compute_dataframe(enriched, snapshot_date=snapshot_date)

    atlas = _atlas_columns_in_order(enriched)
    rollup = aggregate.ma_rollup(atlas.assign(
        ma_ci_low=enriched.get("ma_ci_low"),
        ma_ci_high=enriched.get("ma_ci_high"),
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(atlas, out_dir / "atlas.csv")
    _safe_to_csv(rollup, out_dir / "ma_rollup.csv")
    logger.info("wrote atlas.csv (%d trials) + ma_rollup.csv (%d MAs) to %s",
                len(atlas), len(rollup), out_dir)
    return atlas, rollup


# CSV formula-injection chars per the lessons.md rule. When any of these
# appears as the first char of a string cell, Excel evaluates the cell as a
# formula. Prepend `'` (apostrophe) to neutralise without changing visible
# content. Note: `-` is intentionally NOT in this set per the same lesson —
# negative numbers begin with `-` and we want them parsed as numbers.
_CSV_FORMULA_CHARS = ("=", "+", "@", "\t", "\r")


def _csv_safe(value):
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_CHARS):
        return "'" + value
    return value


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    safe = df.copy()
    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(_csv_safe)
    safe.to_csv(path, index=False, lineterminator="\n")
