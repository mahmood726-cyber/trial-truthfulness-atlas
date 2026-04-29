import os
from pathlib import Path

import pytest

from tta import config


def test_aact_path_default_resolves_to_d_drive():
    assert config.AACT_SNAPSHOT_DIR == Path(r"D:\AACT-storage\AACT\2026-04-12")


def test_pairwise70_path_default_resolves_to_c_projects():
    assert config.PAIRWISE70_DIR == Path(r"C:\Projects\Pairwise70\data")


def test_dossiergap_fixture_path_default():
    assert config.DOSSIERGAP_FIXTURE == Path(
        r"C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv"
    )


def test_n_drift_threshold_default_is_ten_percent():
    assert config.N_DRIFT_THRESHOLD == 0.10


def test_direction_epsilon_default():
    assert config.DIRECTION_EPSILON == 0.01


def test_paths_overridable_via_env(monkeypatch, tmp_path):
    """Reload required because module-level env-var reads are cached at first import."""
    monkeypatch.setenv("TTA_AACT_DIR", str(tmp_path / "aact"))
    import importlib

    importlib.reload(config)
    try:
        assert config.AACT_SNAPSHOT_DIR == tmp_path / "aact"
    finally:
        monkeypatch.delenv("TTA_AACT_DIR")
        importlib.reload(config)
