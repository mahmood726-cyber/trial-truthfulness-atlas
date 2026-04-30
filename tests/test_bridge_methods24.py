"""Tests for bridge methods 2 (PMID cross-ref) and 4 (surname+year heuristic).

Method 2: AACT id_information cross-ref by PMID (~0.95 confidence)
Method 4: surname + year ±1 heuristic match against AACT official_title (~0.65)

Waterfall order: 1 → 2 → 4 → unbridgeable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tta import bridge, ingest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
AACT_DIR = FIXTURES_DIR / "aact_sample"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def dg(fixtures_dir):
    return bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")


@pytest.fixture
def id_info():
    return ingest.load_aact_table(AACT_DIR, "id_information")


@pytest.fixture
def aact_studies():
    return ingest.load_aact_table(AACT_DIR, "studies")


# ---------------------------------------------------------------------------
# Fixture existence / shape
# ---------------------------------------------------------------------------

def test_id_information_fixture_exists():
    assert (AACT_DIR / "id_information.txt").is_file()


def test_id_information_has_required_columns():
    df = ingest.load_aact_table(AACT_DIR, "id_information")
    assert {"nct_id", "id_type", "id_value"} <= set(df.columns)


def test_dossiergap_has_expected_pmid_column(fixtures_dir):
    df = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    assert "expected_pmid" in df.columns


def test_studies_fixture_has_official_title():
    df = ingest.load_aact_table(AACT_DIR, "studies")
    assert "official_title" in df.columns


# ---------------------------------------------------------------------------
# Method 2 — PMID cross-ref
# ---------------------------------------------------------------------------

def test_method2_succeeds_when_pmid_in_id_information(dg, id_info):
    """When method 1 misses but the PMID is in id_information, return the NCT."""
    pw_row = pd.Series({
        "Study": "BridgePMID-Test 2022",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
        "expected_pmid": "30007231",
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=id_info)
    assert result.nct_id == "NCT00001111"
    assert result.method == "pmid_crossref"
    assert result.confidence == pytest.approx(0.95)


def test_method2_falls_through_when_pmid_absent_from_id_information(dg, id_info):
    """PMID not in id_information → method 2 cannot bridge."""
    pw_row = pd.Series({
        "Study": "Mystery Author 2030",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
        "expected_pmid": "99999999",  # not in fixture
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=id_info)
    assert result.nct_id is None
    assert result.method in ("unbridgeable", "surname_year")  # depends on method-4 match


def test_method2_skipped_when_no_pmid_in_row(dg, id_info):
    """If pw_row has no expected_pmid, method 2 is silently skipped."""
    pw_row = pd.Series({
        "Study": "BridgePMID-Test 2022",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
        # no expected_pmid key
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=id_info)
    # Should not find NCT via method 2; may find via method 4 or unbridgeable.
    assert result.method != "pmid_crossref"


def test_method2_skipped_when_aact_id_information_is_none(dg):
    """Passing aact_id_information=None disables method 2 (backwards compat)."""
    pw_row = pd.Series({
        "Study": "BridgePMID-Test 2022",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
        "expected_pmid": "30007231",
    })
    result = bridge.bridge_one(pw_row, dossiergap=dg, aact_id_information=None)
    assert result.method != "pmid_crossref"


# ---------------------------------------------------------------------------
# Method 4 — surname + year heuristic
# ---------------------------------------------------------------------------

def test_method4_succeeds_on_matching_title_and_year(dg, id_info, aact_studies):
    """Parse <Surname> <Year> from Study, match against official_title + year ±1."""
    pw_row = pd.Series({
        "Study": "Sullivan 2019",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.nct_id == "NCT00001111"
    assert result.method == "surname_year"
    assert result.confidence == pytest.approx(0.65)


def test_method4_respects_plus_minus_one_year_window(dg, id_info, aact_studies):
    """Year ±1: a Study with year 2018 should still match a 2019 trial."""
    pw_row = pd.Series({
        "Study": "Sullivan 2018",  # 2018; fixture trial is 2019 → within ±1
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.nct_id == "NCT00001111"
    assert result.method == "surname_year"


def test_method4_fails_when_year_out_of_window(dg, id_info, aact_studies):
    """Year 2015 does not match a 2019 trial (gap = 4 years)."""
    pw_row = pd.Series({
        "Study": "Sullivan 2015",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.nct_id != "NCT00001111" or result.method != "surname_year"


def test_method4_fails_when_surname_not_in_title(dg, id_info, aact_studies):
    """Surname not present in any official_title → method 4 misses."""
    pw_row = pd.Series({
        "Study": "Zephyrus 2019",  # surname not in any fixture title
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.nct_id is None
    assert result.method == "unbridgeable"


def test_method4_skipped_when_aact_studies_is_none(dg, id_info):
    """Passing aact_studies=None disables method 4."""
    pw_row = pd.Series({
        "Study": "Sullivan 2019",
        "review_doi": "10.1002/14651858.CDFAKE_NOEXIST.pub1",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=None,
    )
    assert result.method != "surname_year"


def test_method4_skipped_without_official_title_column(dg, id_info):
    """If aact_studies has no official_title column, method 4 is silently skipped."""
    studies_no_title = pd.DataFrame([{"nct_id": "NCT00001111", "completion_date": "2019-12-01"}])
    pw_row = pd.Series({"Study": "Sullivan 2019", "review_doi": "10.1002/xyz"})
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=studies_no_title,
    )
    assert result.method != "surname_year"


# ---------------------------------------------------------------------------
# Waterfall: method 1 beats method 2 beats method 4
# ---------------------------------------------------------------------------

def test_method1_takes_priority_over_method2(dg, id_info, aact_studies):
    """When method 1 succeeds, methods 2 and 4 are not tried."""
    pw_row = pd.Series({
        "Study": "McMurray 2014",
        "review_doi": "10.1002/14651858.CD012612.pub2",
        "expected_pmid": "30007231",  # would succeed via method 2
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.method == "dossiergap_direct"
    assert result.nct_id == "NCT01035255"


def test_all_methods_miss_returns_unbridgeable(dg, id_info, aact_studies):
    """When all methods miss, result is unbridgeable with confidence 0.0."""
    pw_row = pd.Series({
        "Study": "Zephyrus 2050",
        "review_doi": "10.1002/nobody",
    })
    result = bridge.bridge_one(
        pw_row, dossiergap=dg, aact_id_information=id_info,
        aact_studies=aact_studies,
    )
    assert result.nct_id is None
    assert result.method == "unbridgeable"
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# bridge_pairwise70 wires expected_pmid from DossierGap
# ---------------------------------------------------------------------------

def test_bridge_pairwise70_wires_pmid_from_dossiergap(fixtures_dir, id_info, aact_studies):
    """bridge_pairwise70 should add expected_pmid from DossierGap to each row."""
    from tta import ingest
    dg_with_pmid = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    bridged = bridge.bridge_pairwise70(
        pw, dossiergap=dg_with_pmid,
        aact_id_information=id_info, aact_studies=aact_studies,
    )
    # McMurray 2014 is in DossierGap with expected_pmid; must be present in bridged.
    mcmurray = bridged[bridged["Study"] == "McMurray 2014"]
    assert not mcmurray.empty
    assert pd.notna(mcmurray["expected_pmid"].iloc[0])
