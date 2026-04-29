from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tta import pipeline


@pytest.fixture
def stub_ollama():
    """Returns labels in fixture order — only used on cache miss."""
    client = MagicMock()
    client.get_model_version.return_value = "gemma2:9b@stub_for_test"
    client.classify.side_effect = [
        "identical",                # PARADIGM-HF
        "refinement",               # VICTORIA
        "substantively_different",  # GRIPHON
        "identical",                # FIXTURE-N
        # Trial 5 is unscoreable upstream, no LLM call
    ]
    return client


def test_5_trial_pipeline_produces_pinned_atlas(tmp_path, fixtures_dir, stub_ollama):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    atlas, ma_rollup = pipeline.run_5trial_fixture(
        fixtures_dir=fixtures_dir,
        out_dir=out_dir,
        snapshot_date=date(2026, 4, 12),
        ollama_client=stub_ollama,
    )

    expected_atlas = pd.read_csv(fixtures_dir / "expected" / "atlas.csv")
    expected_rollup = pd.read_csv(fixtures_dir / "expected" / "ma_rollup.csv")

    pd.testing.assert_frame_equal(
        atlas.reset_index(drop=True),
        expected_atlas.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        ma_rollup.reset_index(drop=True),
        expected_rollup.reset_index(drop=True),
        check_dtype=False,
    )


def test_5_trial_pipeline_writes_csv_files(tmp_path, fixtures_dir, stub_ollama):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pipeline.run_5trial_fixture(
        fixtures_dir=fixtures_dir,
        out_dir=out_dir,
        snapshot_date=date(2026, 4, 12),
        ollama_client=stub_ollama,
    )
    assert (out_dir / "atlas.csv").exists()
    assert (out_dir / "ma_rollup.csv").exists()


def test_5_trial_pipeline_handles_empty_input(tmp_path, fixtures_dir, stub_ollama):
    """Per Sentinel P1-empty-dataframe-access: empty input must not crash."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    empty_pw = tmp_path / "empty_pw70"
    empty_pw.mkdir()
    src_rda = fixtures_dir / "pairwise70_sample" / "CDFAKE002_pub1_data.rda"
    shutil.copy(src_rda, empty_pw / "CDFAKE002_pub1_data.rda")
    atlas, rollup = pipeline.run_5trial_fixture(
        fixtures_dir=fixtures_dir,
        out_dir=out_dir,
        snapshot_date=date(2026, 4, 12),
        ollama_client=stub_ollama,
        pairwise70_dir_override=empty_pw,
    )
    assert isinstance(atlas, pd.DataFrame)
    assert isinstance(rollup, pd.DataFrame)
