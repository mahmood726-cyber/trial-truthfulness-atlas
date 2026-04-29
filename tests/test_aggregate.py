from __future__ import annotations

import pandas as pd
import pytest

from tta import aggregate


@pytest.fixture
def sample_atlas() -> pd.DataFrame:
    return pd.DataFrame({
        "nct_id": ["NCT01", "NCT02", "NCT03", "NCT04", "NCT05"],
        "review_id": ["CD001_pub1", "CD001_pub1", "CD001_pub1", "CD002_pub1", "CD002_pub1"],
        "review_doi": ["10/CD001"] * 3 + ["10/CD002"] * 2,
        "bridge_method": ["dossiergap_direct", "dossiergap_direct", "unbridgeable",
                          "dossiergap_direct", "dossiergap_direct"],
        "outcome_drift": ["identical", "refinement", "unscoreable",
                          "substantively_different", "identical"],
        "n_drift": ["not_flagged", "flagged", "unscoreable", "not_flagged", "not_flagged"],
        "direction_concordance": ["concordant", "concordant", "unscoreable",
                                  "flipped", "concordant"],
        "results_posting": ["posted", "required_not_posted", "unscoreable",
                            "posted", "not_required"],
        "ma_effect_log": [-0.2, -0.1, None, 0.05, -0.3],
    })


def test_rollup_one_row_per_review(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    assert len(out) == 2
    assert set(out["review_id"]) == {"CD001_pub1", "CD002_pub1"}


def test_rollup_n_trials_counts_all_rows(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_trials"] == 3
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    assert cd002["n_trials"] == 2


def test_rollup_n_unbridgeable(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_unbridgeable"] == 1


def test_rollup_any_flag_count(sample_atlas):
    # Trial counts as flagged if ANY of the 4 truthfulness flags is in {flagged,
    # substantively_different, flipped, required_not_posted}
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    # NCT02 flagged on n_drift + required_not_posted => 1 trial flagged
    assert cd001["n_trials_with_any_flag"] == 1
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    # NCT04 has substantively_different + flipped => 1 trial flagged
    assert cd002["n_trials_with_any_flag"] == 1


def test_rollup_per_flag_counts(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_outcome_drift_substantive"] == 0
    assert cd001["n_n_drift_flagged"] == 1
    assert cd001["n_direction_flipped"] == 0
    assert cd001["n_results_required_not_posted"] == 1


def test_rollup_crosses_null_when_ci_includes_zero(sample_atlas):
    # v0.1.x convention: crosses_null comes from the FIRST non-null trial CI
    # in the group (proper per-MA pooled CI is v0.2.0 work).
    df = sample_atlas.copy()
    df["ma_ci_low"] = [-0.3, -0.2, None, -0.05, -0.4]
    df["ma_ci_high"] = [-0.1, 0.0, None, 0.15, -0.2]
    out = aggregate.ma_rollup(df)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    # CD001 first row CI = [-0.3, -0.1] — does NOT include 0.0 → False.
    # CD002 first row CI = [-0.05, 0.15] — DOES include 0.0 → True.
    # Pandas may store these as numpy.bool_; compare by value not identity.
    assert bool(cd001["crosses_null"]) is False
    assert bool(cd002["crosses_null"]) is True


def test_rollup_handles_truly_empty_dataframe():
    """Per Sentinel P1-empty-dataframe-access regression: ma_rollup must
    not crash on a zero-row input. v0.1.0 test covered the wrong condition
    (a non-empty MA group) — this exercises the actual early-return path."""
    empty = pd.DataFrame(columns=[
        "review_id", "review_doi", "bridge_method", "outcome_drift",
        "n_drift", "direction_concordance", "results_posting",
    ])
    out = aggregate.ma_rollup(empty)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    # Schema preserved: columns match the populated-output contract.
    assert "n_trials" in out.columns
    assert "crosses_null" in out.columns
