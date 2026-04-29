from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tta.flags import results_posting as rp


SNAPSHOT = date(2026, 4, 12)


def test_posted_when_results_date_present():
    assert rp.classify(
        results_first_posted_date=date(2015, 1, 9),
        completion_date=date(2014, 3, 31),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "posted"


def test_required_not_posted_when_completed_long_ago_and_no_results():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "required_not_posted"


def test_not_required_when_observational():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="observational",
        intervention_types=[],
        snapshot_date=SNAPSHOT,
    ) == "not_required"


def test_not_required_when_only_behavioural_intervention():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="interventional",
        intervention_types=["behavioral"],
        snapshot_date=SNAPSHOT,
    ) == "not_required"


def test_unscoreable_when_completion_date_missing():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=None,
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "unscoreable"


def test_unscoreable_when_completed_recently():
    # within 12 mo of snapshot: not yet required
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2025, 6, 1),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "unscoreable"


def test_compute_dataframe(fixtures_dir):
    df = pd.DataFrame({
        "nct_id": ["NCT01035255", "NCT99999001", "NCT99999009"],
        "results_first_posted_date": [date(2015, 1, 9), None, None],
        "completion_date": [date(2014, 3, 31), date(2018, 6, 1), date(2019, 1, 1)],
        "study_type": ["interventional", "interventional", "observational"],
        "intervention_types": [["drug"], ["drug"], ["behavioral"]],
    })
    out = rp.compute_dataframe(df, snapshot_date=SNAPSHOT)
    assert list(out["results_posting"]) == ["posted", "required_not_posted", "not_required"]
