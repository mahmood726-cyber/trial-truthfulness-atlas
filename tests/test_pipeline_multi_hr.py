"""Tests for P1-16: multi-HR result_groups join via outcome_id.

AACT has multiple outcome_analyses rows per trial, one per outcome.
_enrich_with_aact must join to the outcomes table and select the row
whose outcome_type == 'Primary', falling back to the first available row
when no Primary row exists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tta import pipeline, ingest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AACT_DIR = FIXTURES_DIR / "aact_sample"


@pytest.fixture
def outcomes():
    return ingest.load_aact_table(AACT_DIR, "outcomes")


@pytest.fixture
def outcome_analyses():
    return ingest.load_aact_table(AACT_DIR, "outcome_analyses")


@pytest.fixture
def studies():
    return ingest.load_aact_table(AACT_DIR, "studies")


@pytest.fixture
def design_outcomes():
    return ingest.load_aact_table(AACT_DIR, "design_outcomes")


@pytest.fixture
def calculated():
    return ingest.load_aact_table(AACT_DIR, "calculated_values")


@pytest.fixture
def interventions():
    return ingest.load_aact_table(AACT_DIR, "interventions")


def test_outcomes_fixture_exists():
    """Fixture file must exist."""
    assert (AACT_DIR / "outcomes.txt").is_file()


def test_outcomes_fixture_has_required_columns():
    df = ingest.load_aact_table(AACT_DIR, "outcomes")
    assert {"id", "nct_id", "outcome_type"} <= set(df.columns)


def test_outcome_analyses_has_outcome_id_column():
    """outcome_analyses must have outcome_id for the join."""
    df = ingest.load_aact_table(AACT_DIR, "outcome_analyses")
    assert "outcome_id" in df.columns


def test_outcome_analyses_has_two_rows_per_nct():
    """Each NCT with results should have >=2 HR rows (primary + secondary)."""
    df = ingest.load_aact_table(AACT_DIR, "outcome_analyses")
    hr = df[df["param_type"] == "Hazard Ratio"]
    counts = hr.groupby("nct_id").size()
    # At least one NCT must have 2 rows to prove the multi-HR path is exercised.
    assert (counts >= 2).any(), "expected at least one NCT with 2 HR rows in fixture"


def _make_bridged_row(nct_id: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "nct_id": nct_id,
        "Study": "Test 2020",
        "review_id": "TEST001",
        "review_doi": "10.1002/TEST",
        "bridge_method": "dossiergap_direct",
        "bridge_confidence": 0.99,
        "ma_n": 100,
        "ma_extracted_outcome": "Primary composite",
        "ma_effect_log": -0.1,
        "ma_ci_low": -0.2,
        "ma_ci_high": 0.0,
    }])


def test_primary_hr_is_selected_over_secondary(
    studies, design_outcomes, calculated, outcome_analyses, outcomes, interventions
):
    """When a trial has both a Primary and a Secondary HR row, the Primary is picked."""
    # NCT01035255 has two HR rows in the updated fixture: one Primary, one Secondary.
    bridged = _make_bridged_row("NCT01035255")
    enriched = pipeline._enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions, outcomes=outcomes,
    )
    primary_hr = outcomes[
        (outcomes["nct_id"] == "NCT01035255") & (outcomes["outcome_type"] == "Primary")
    ]["id"].iloc[0]
    expected_hr = float(
        outcome_analyses[outcome_analyses["outcome_id"] == primary_hr]["param_value"].iloc[0]
    )
    import math
    assert enriched["registered_effect_log"].iloc[0] == pytest.approx(math.log(expected_hr))


def test_secondary_hr_not_selected_when_primary_exists(
    studies, design_outcomes, calculated, outcome_analyses, outcomes, interventions
):
    """The secondary-outcome HR must NOT appear when a primary-outcome HR exists."""
    bridged = _make_bridged_row("NCT01035255")
    enriched = pipeline._enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions, outcomes=outcomes,
    )
    secondary_hr = float(
        outcome_analyses[
            (outcome_analyses["nct_id"] == "NCT01035255") &
            (outcome_analyses["outcome_id"].isin(
                outcomes[
                    (outcomes["nct_id"] == "NCT01035255") &
                    (outcomes["outcome_type"] == "Secondary")
                ]["id"]
            ))
        ]["param_value"].iloc[0]
    )
    import math
    # The registered_effect_log must NOT equal log(secondary_hr)
    assert enriched["registered_effect_log"].iloc[0] != pytest.approx(math.log(secondary_hr))


def test_fallback_to_first_row_when_no_primary(
    studies, design_outcomes, calculated, outcome_analyses, outcomes, interventions
):
    """When no Primary outcome row exists, fall back to the first available HR row."""
    # NCT99999001 has only a Secondary HR row in the updated fixture (no Primary HR).
    bridged = _make_bridged_row("NCT99999001")
    enriched = pipeline._enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions, outcomes=outcomes,
    )
    # Should still produce a non-null registered_effect_log (fallback used).
    assert pd.notna(enriched["registered_effect_log"].iloc[0])


def test_enrich_with_aact_backwards_compat_no_outcomes_arg(
    studies, design_outcomes, calculated, outcome_analyses, interventions
):
    """Calling _enrich_with_aact without the outcomes kwarg (old call-site) still works."""
    bridged = _make_bridged_row("NCT01035255")
    enriched = pipeline._enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions,
    )
    # Without outcomes, falls back to the old drop_duplicates behaviour.
    assert "registered_effect_log" in enriched.columns
