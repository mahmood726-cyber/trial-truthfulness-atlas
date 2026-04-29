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


def test_rollup_crosses_null_uses_pooled_ci(sample_atlas):
    """v0.2.0: crosses_null derives from the random-effects POOLED CI,
    not the first trial's CI (P1-1 fix from multi-persona review)."""
    df = sample_atlas.copy()
    df["ma_ci_low"] = [-0.3, -0.2, None, -0.05, -0.4]
    df["ma_ci_high"] = [-0.1, 0.0, None, 0.15, -0.2]
    out = aggregate.ma_rollup(df)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    # CD001: 2 valid trials (3rd has None CI); k=2 → t_crit = 12.706 at df=1
    # widens the pooled HKSJ CI substantially → expect True (crosses null).
    assert bool(cd001["crosses_null"]) is True
    # CD002: 2 trials with disagreeing effects (0.05, -0.3); pool spans 0.
    assert bool(cd002["crosses_null"]) is True
    # Both rollups should have the new pooled-effect columns populated.
    assert cd001["pool_method"] == "random_effects_dl_hksj"
    assert cd002["pool_method"] == "random_effects_dl_hksj"
    assert cd001["tau2"] is not None
    assert cd001["pooled_effect_log"] is not None


def test_rollup_pooled_effect_columns_present(sample_atlas):
    """v0.2.0 schema: ma_rollup output includes pooled_effect_log,
    pooled_ci_low, pooled_ci_high, tau2, pool_method."""
    df = sample_atlas.copy()
    df["ma_ci_low"] = [-0.3, -0.2, None, -0.05, -0.4]
    df["ma_ci_high"] = [-0.1, 0.0, None, 0.15, -0.2]
    out = aggregate.ma_rollup(df)
    for col in ["pooled_effect_log", "pooled_ci_low", "pooled_ci_high",
                "tau2", "pool_method"]:
        assert col in out.columns, f"missing column {col}"


def test_rollup_pool_method_no_trials_when_effects_missing():
    """If all rows in a group have NaN effects, pool falls back to no_trials."""
    df = pd.DataFrame({
        "nct_id": ["NCT01", "NCT02"],
        "review_id": ["CD_X", "CD_X"],
        "review_doi": ["10/CD_X", "10/CD_X"],
        "bridge_method": ["unbridgeable", "unbridgeable"],
        "outcome_drift": ["unscoreable", "unscoreable"],
        "n_drift": ["unscoreable", "unscoreable"],
        "direction_concordance": ["unscoreable", "unscoreable"],
        "results_posting": ["unscoreable", "unscoreable"],
        "ma_effect_log": [None, None],
    })
    out = aggregate.ma_rollup(df)
    row = out.iloc[0]
    assert row["pool_method"] == "no_trials"
    assert row["pooled_effect_log"] is None
    assert row["crosses_null"] is None


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
