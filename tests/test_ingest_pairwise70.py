from __future__ import annotations

import pandas as pd

from tta import ingest


def test_load_pairwise70_one_file(fixtures_dir):
    df = ingest.load_pairwise70_rda(
        fixtures_dir / "pairwise70_sample" / "CDFAKE001_pub1_data.rda"
    )
    assert "Study" in df.columns
    assert "review_doi" in df.columns
    assert len(df) == 3
    assert df.attrs["review_id"] == "CDFAKE001_pub1"


def test_load_all_pairwise70_in_dir(fixtures_dir):
    df = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    assert len(df) == 5
    assert "review_id" in df.columns
    assert set(df["review_id"]) == {"CDFAKE001_pub1", "CDFAKE002_pub1"}


def test_pairwise70_to_parquet(fixtures_dir, tmp_path):
    out = ingest.materialise_pairwise70(
        fixtures_dir / "pairwise70_sample",
        tmp_path / "pw70.parquet",
    )
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 5
