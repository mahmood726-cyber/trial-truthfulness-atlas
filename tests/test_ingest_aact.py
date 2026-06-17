from __future__ import annotations

import pandas as pd
import pytest

from tta import ingest


REQUIRED_TABLES = [
    "studies",
    "design_outcomes",
    "calculated_values",
    "outcome_analyses",
    "interventions",
    "browse_conditions",
]


def test_load_aact_tsv_returns_dataframe(fixtures_dir):
    df = ingest.load_aact_table(fixtures_dir / "aact_sample", "studies")
    assert isinstance(df, pd.DataFrame)
    assert "nct_id" in df.columns
    assert len(df) == 6


def test_load_aact_pipe_separator(fixtures_dir):
    df = ingest.load_aact_table(fixtures_dir / "aact_sample", "design_outcomes")
    assert set(df["outcome_type"].unique()) == {"primary"}


def test_load_aact_blank_dates_become_nat(fixtures_dir):
    df = ingest.load_aact_table(fixtures_dir / "aact_sample", "studies")
    assert pd.isna(df.loc[df["nct_id"] == "NCT99999001", "results_first_posted_date"].iloc[0])
    assert pd.notna(df.loc[df["nct_id"] == "NCT01035255", "results_first_posted_date"].iloc[0])


def test_materialise_to_parquet(fixtures_dir, tmp_path):
    out = ingest.materialise_aact(
        fixtures_dir / "aact_sample",
        tmp_path,
        tables=REQUIRED_TABLES,
    )
    for t in REQUIRED_TABLES:
        assert (tmp_path / f"aact_{t}.parquet").exists()
        df = pd.read_parquet(tmp_path / f"aact_{t}.parquet")
        assert len(df) > 0
    assert set(out.keys()) == set(REQUIRED_TABLES)


def test_load_aact_missing_table_raises(fixtures_dir):
    with pytest.raises(FileNotFoundError):
        ingest.load_aact_table(fixtures_dir / "aact_sample", "does_not_exist")
