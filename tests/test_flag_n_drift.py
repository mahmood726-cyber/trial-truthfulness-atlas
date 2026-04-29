from __future__ import annotations

import pandas as pd
import pytest

from tta.flags import n_drift


def test_below_threshold_not_flagged():
    assert n_drift.classify(registered_n=1000, ma_n=950, threshold=0.10) == "not_flagged"


def test_above_threshold_flagged():
    assert n_drift.classify(registered_n=1000, ma_n=700, threshold=0.10) == "flagged"


def test_exact_threshold_boundary_not_flagged():
    # 100/1000 = 0.10 exactly → boundary — we treat "not_flagged"
    assert n_drift.classify(registered_n=1000, ma_n=900, threshold=0.10) == "not_flagged"


def test_missing_registered_unscoreable():
    assert n_drift.classify(registered_n=None, ma_n=500, threshold=0.10) == "unscoreable"


def test_missing_ma_unscoreable():
    assert n_drift.classify(registered_n=500, ma_n=None, threshold=0.10) == "unscoreable"


def test_zero_registered_unscoreable():
    assert n_drift.classify(registered_n=0, ma_n=500, threshold=0.10) == "unscoreable"


def test_negation_scrubber_drops_not_randomized():
    text = "Enrolled 5050. Not Randomized 1807. Analysed 5050."
    found = n_drift.extract_first_n(text)
    assert found == 5050  # the 1807 must NOT win


def test_negation_scrubber_drops_non_randomized():
    text = "Total 1200 patients; non-randomized observational arm 300."
    found = n_drift.extract_first_n(text)
    assert found == 1200


def test_extract_returns_none_when_no_numbers():
    assert n_drift.extract_first_n("no numbers here") is None


def test_compute_dataframe_uses_threshold_from_config():
    df = pd.DataFrame({
        "nct_id": ["a", "b", "c", "d"],
        "registered_n": [1000, 1000, None, 1000],
        "ma_n": [950, 700, 500, None],
    })
    out = n_drift.compute_dataframe(df, threshold=0.10)
    assert list(out["n_drift"]) == ["not_flagged", "flagged", "unscoreable", "unscoreable"]
