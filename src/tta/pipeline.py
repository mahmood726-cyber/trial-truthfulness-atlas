"""End-to-end pipeline runner for the v0.1.0 fixture and (later) full sweep."""

from __future__ import annotations

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


def _enrich_with_aact(
    bridged: pd.DataFrame,
    aact_studies: pd.DataFrame,
    aact_design_outcomes: pd.DataFrame,
    aact_calculated: pd.DataFrame,
    aact_outcome_analyses: pd.DataFrame,
    aact_interventions: pd.DataFrame,
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

    hr = (
        aact_outcome_analyses[aact_outcome_analyses["param_type"] == "Hazard Ratio"]
        .drop_duplicates("nct_id")
        .set_index("nct_id")["param_value"]
        .astype(float)
    )
    out["registered_effect_log"] = out["nct_id"].map(
        lambda nct: math.log(hr[nct]) if nct in hr.index and not pd.isna(hr[nct]) else None
    )

    studies = aact_studies.drop_duplicates("nct_id").set_index("nct_id")
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
    cols = [
        "nct_id", "review_id", "review_doi", "Study",
        "bridge_method", "bridge_confidence",
        "registered_outcome", "ma_extracted_outcome", "outcome_drift",
        "registered_n", "ma_n", "n_drift",
        "registered_effect_log", "ma_effect_log", "direction_concordance",
        "results_posting",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def run_5trial_fixture(
    fixtures_dir: Path,
    out_dir: Path,
    snapshot_date: date,
    ollama_client,
    pairwise70_dir_override: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aact_dir = fixtures_dir / "aact_sample"
    pw_dir = pairwise70_dir_override or (fixtures_dir / "pairwise70_sample")

    pw = ingest.load_pairwise70_dir(pw_dir)
    pw = cardio_filter.filter_pairwise70(pw)
    pw["ma_n"] = pw.get("Experimental.N", 0).fillna(0).astype(int) + pw.get("Control.N", 0).fillna(0).astype(int)
    pw["ma_extracted_outcome"] = pw.get("Analysis.name")
    pw["ma_effect_log"] = pw.get("Mean")
    pw["ma_ci_low"] = pw.get("CI.start")
    pw["ma_ci_high"] = pw.get("CI.end")

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    bridged = bridge.bridge_pairwise70(pw, dossiergap=dg, aact_id_information=None)

    studies = ingest.load_aact_table(aact_dir, "studies")
    design_outcomes = ingest.load_aact_table(aact_dir, "design_outcomes")
    calculated = ingest.load_aact_table(aact_dir, "calculated_values")
    outcome_analyses = ingest.load_aact_table(aact_dir, "outcome_analyses")
    interventions = ingest.load_aact_table(aact_dir, "interventions")

    enriched = _enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions,
    )

    # Convert NaN in string/object columns to None so downstream guards fire correctly.
    # pandas NaN is truthy; Python None is falsy — flag compute_one guards use `not x`.
    for col in ("registered_outcome", "ma_extracted_outcome", "study_type"):
        if col in enriched.columns:
            enriched[col] = enriched[col].where(pd.notna(enriched[col]), other=None)

    judge_cache = cache_mod.JudgeCache(out_dir / "judge_cache")
    enriched = outcome_drift.compute_dataframe(enriched, client=ollama_client, cache=judge_cache)
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
    atlas.to_csv(out_dir / "atlas.csv", index=False, lineterminator="\n")
    rollup.to_csv(out_dir / "ma_rollup.csv", index=False, lineterminator="\n")
    return atlas, rollup
