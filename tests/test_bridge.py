from __future__ import annotations

import pandas as pd
import pytest

from tta import bridge


def test_load_dossiergap_returns_normalised_columns(fixtures_dir):
    df = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    assert {"nct_id", "study_label", "review_doi"} <= set(df.columns)
    assert len(df) == 4


def test_bridge_direct_match_succeeds(fixtures_dir):
    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw_row = pd.Series({
        "Study": "McMurray 2014",
        "review_doi": "10.1002/14651858.CD012612.pub2",
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=None)
    assert result.nct_id == "NCT01035255"
    assert result.method == "dossiergap_direct"
    assert result.confidence >= 0.95


def test_bridge_unbridgeable_when_no_match(fixtures_dir):
    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw_row = pd.Series({
        "Study": "Unknown Author 1999",
        "review_doi": "10.1002/14651858.CDFAKE002.pub1",
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=None)
    assert result.nct_id is None
    assert result.method == "unbridgeable"
    assert result.confidence == 0.0


def test_bridge_dataframe_marks_each_row(fixtures_dir):
    from tta import ingest

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    bridged = bridge.bridge_pairwise70(pw, dossiergap=dg, aact_id_information=None)
    assert len(bridged) == len(pw)
    assert "nct_id" in bridged.columns
    assert "bridge_method" in bridged.columns
    assert "bridge_confidence" in bridged.columns

    unknown = bridged[bridged["Study"] == "Unknown Author 1999"]
    assert unknown["bridge_method"].iloc[0] == "unbridgeable"

    paradigm = bridged[bridged["Study"] == "McMurray 2014"]
    assert paradigm["nct_id"].iloc[0] == "NCT01035255"


def test_bridge_resolution_rate(fixtures_dir):
    from tta import ingest

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    bridged = bridge.bridge_pairwise70(pw, dossiergap=dg, aact_id_information=None)
    rate = bridge.resolution_rate(bridged)
    assert rate == pytest.approx(0.80, abs=1e-9)
