from __future__ import annotations

import pandas as pd
import pytest

from tta.flags import direction_concordance as dc


def test_same_sign_concordant():
    assert dc.classify(registered_effect=-0.2, ma_effect=-0.1) == "concordant"
    assert dc.classify(registered_effect=0.5, ma_effect=0.3) == "concordant"


def test_opposite_sign_flipped():
    assert dc.classify(registered_effect=-0.2, ma_effect=0.1) == "flipped"
    assert dc.classify(registered_effect=0.3, ma_effect=-0.05) == "flipped"


def test_near_zero_unscoreable():
    assert dc.classify(registered_effect=0.005, ma_effect=-0.5) == "unscoreable"
    assert dc.classify(registered_effect=-0.5, ma_effect=0.005) == "unscoreable"


def test_missing_unscoreable():
    assert dc.classify(registered_effect=None, ma_effect=-0.1) == "unscoreable"
    assert dc.classify(registered_effect=-0.1, ma_effect=None) == "unscoreable"


def test_hr_to_log_scale():
    # AACT param_value for HR is on natural scale; we convert to log.
    # HR < 1 means protective ⇒ log(HR) < 0
    assert dc.hr_to_log_effect(0.80) < 0
    assert dc.hr_to_log_effect(1.20) > 0
    assert dc.hr_to_log_effect(1.00) == pytest.approx(0.0)


def test_compute_dataframe_with_aact_join():
    df = pd.DataFrame({
        "nct_id": ["a", "b", "c", "d"],
        "registered_effect_log": [-0.22, -0.10, 0.005, None],
        "ma_effect_log": [-0.20, 0.15, -0.5, -0.1],
    })
    out = dc.compute_dataframe(df)
    assert list(out["direction_concordance"]) == [
        "concordant", "flipped", "unscoreable", "unscoreable",
    ]
