from __future__ import annotations

import re

import pandas as pd
import pytest

from tta.dashboards import verification_ui


@pytest.fixture
def sample_atlas():
    return pd.DataFrame({
        "nct_id": ["NCT01035255", "NCT02861534"],
        "review_id": ["CDFAKE001_pub1", "CDFAKE001_pub1"],
        "review_doi": ["10.1002/14651858.CD012612.pub2", "10.1002/14651858.CD013650.pub2"],
        "Study": ["McMurray 2014", "Armstrong 2020"],
        "bridge_method": ["dossiergap_direct", "dossiergap_direct"],
        "registered_outcome": ["Composite endpoint of CV death and HF hospitalisation",
                                "Time to first occurrence of CV death or HF hospitalisation"],
        "ma_extracted_outcome": ["CV death or HF hospitalisation",
                                  "CV death or HF hospitalisation"],
        "outcome_drift": ["identical", "refinement"],
        "registered_n": [8442, 5050],
        "ma_n": [8442, 5050],
        "n_drift": ["not_flagged", "not_flagged"],
        "registered_effect_log": [-0.223, -0.105],
        "ma_effect_log": [-0.223, -0.105],
        "direction_concordance": ["concordant", "concordant"],
        "results_posting": ["posted", "posted"],
    })


def test_render_returns_full_html(sample_atlas):
    out = verification_ui.render(sample_atlas, title="TTA verify")
    assert "<!doctype html>" in out.lower()
    assert "</html>" in out


def test_render_inlines_no_external_resources(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "http://" not in out
    assert "<script src=" not in out
    assert "<link" not in out


def test_render_uses_unique_localstorage_key(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "tta-verification-v0.1.0" in out


def test_render_one_trial_at_a_time(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "NCT01035255" in out
    assert "NCT02861534" in out
    assert "data-trial-index" in out
    assert "Next" in out
    assert "Previous" in out


def test_render_has_confirm_disagree_skip_controls(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "Confirm" in out
    assert "Disagree" in out
    assert "Skip" in out


def test_render_has_export_button(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "Export verifications" in out or "downloadJson" in out


def test_render_has_no_literal_close_script_in_template(sample_atlas):
    """Per lessons.md JS rule: no literal </script> inside <script>."""
    out = verification_ui.render(sample_atlas, title="x")
    opens = out.count("<script>") + out.count('<script ')
    closes = out.count("</script>")
    assert opens == closes


def test_render_balanced_divs(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    open_div = len(re.findall(r"<div\b", out))
    close_div = len(re.findall(r"</div>", out))
    assert open_div == close_div
