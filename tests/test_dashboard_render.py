from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from tta.dashboards import atlas_dashboard


@pytest.fixture
def sample_atlas():
    return pd.DataFrame({
        "nct_id": ["NCT01", "NCT02", None],
        "review_id": ["CD001", "CD001", "CD002"],
        "review_doi": ["10/CD001", "10/CD001", "10/CD002"],
        "Study": ["A 2010", "B 2015", "C 2020"],
        "bridge_method": ["dossiergap_direct", "dossiergap_direct", "unbridgeable"],
        "outcome_drift": ["identical", "refinement", "unscoreable"],
        "n_drift": ["not_flagged", "flagged", "unscoreable"],
        "direction_concordance": ["concordant", "concordant", "unscoreable"],
        "results_posting": ["posted", "required_not_posted", "unscoreable"],
    })


def test_render_returns_full_html(sample_atlas, tmp_path):
    out = atlas_dashboard.render(sample_atlas, title="TTA v0.1.0 — fixture")
    assert out.startswith("<!doctype html>") or out.startswith("<!DOCTYPE html>")
    assert "</html>" in out
    assert "TTA v0.1.0" in out


def test_render_inlines_no_external_resources(sample_atlas):
    out = atlas_dashboard.render(sample_atlas, title="x")
    assert "http://" not in out
    assert "https://" not in out or "https://www.w3.org" in out
    assert "<link" not in out


def test_render_contains_one_row_per_trial(sample_atlas):
    out = atlas_dashboard.render(sample_atlas, title="x")
    for nct in ["NCT01", "NCT02"]:
        assert nct in out


def test_render_contains_per_flag_summary(sample_atlas):
    out = atlas_dashboard.render(sample_atlas, title="x")
    assert "Outcome drift" in out
    assert "N drift" in out
    assert "Direction concordance" in out
    assert "Results posting" in out


def test_render_balanced_html(sample_atlas):
    out = atlas_dashboard.render(sample_atlas, title="x")
    open_div = len(re.findall(r"<div\b", out))
    close_div = len(re.findall(r"</div>", out))
    assert open_div == close_div, f"div imbalance: {open_div} vs {close_div}"


def test_write_dashboard(sample_atlas, tmp_path):
    out_path = tmp_path / "dashboard.html"
    atlas_dashboard.write(sample_atlas, out_path, title="x")
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "</html>" in content
