from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tta import cli


def test_parser_recognises_all_subcommands():
    p = cli.build_parser()
    for sub in ["preflight", "build", "sweep", "verify-one"]:
        ns = p.parse_args([sub] + (["--nct", "NCT01"] if sub == "verify-one" else []))
        assert ns.cmd == sub


def test_build_calls_pipeline_with_fixtures(tmp_path, fixtures_dir, monkeypatch):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(cli, "_resolve_fixtures_dir", lambda: fixtures_dir)
    monkeypatch.setattr(cli, "_resolve_out_dir", lambda: out_dir)
    monkeypatch.setattr(cli, "_make_ollama_client", lambda: _stub_client())
    rc = cli.main(["build", "--fixture-mode"])
    assert rc == 0
    assert (out_dir / "atlas.csv").exists()
    assert (out_dir / "ma_rollup.csv").exists()
    assert (out_dir / "dashboard.html").exists()
    assert (out_dir / "verification.html").exists()


def test_verify_one_prints_flags_for_known_nct(tmp_path, fixtures_dir, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_resolve_fixtures_dir", lambda: fixtures_dir)
    monkeypatch.setattr(cli, "_resolve_out_dir", lambda: tmp_path / "out")
    monkeypatch.setattr(cli, "_make_ollama_client", lambda: _stub_client())
    rc = cli.main(["verify-one", "--nct", "NCT01035255", "--fixture-mode"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NCT01035255" in out
    assert "outcome_drift" in out


def _stub_client():
    c = MagicMock()
    c.get_model_version.return_value = "gemma2:9b@stub_for_test"
    # Pipeline-processing order (v0.1.1+): PARADIGM, VICTORIA, FIXTURE-N, GRIPHON.
    c.classify.side_effect = ["identical", "refinement", "identical",
                              "substantively_different"]
    return c


def test_snapshot_date_config_default_used_when_no_arg(monkeypatch):
    """v0.1.2: snapshot date defaults to config.SNAPSHOT_DATE; CLI arg overrides."""
    from tta import config as config_mod

    monkeypatch.setattr(config_mod, "SNAPSHOT_DATE", "2027-09-15")
    p = cli.build_parser()
    ns = p.parse_args(["build", "--fixture-mode"])
    resolved = cli._resolve_snapshot_date(ns)
    assert resolved == date(2027, 9, 15)


def test_snapshot_date_cli_arg_overrides_config(monkeypatch):
    from tta import config as config_mod

    monkeypatch.setattr(config_mod, "SNAPSHOT_DATE", "2027-09-15")
    p = cli.build_parser()
    ns = p.parse_args(["build", "--fixture-mode", "--snapshot-date", "2030-01-31"])
    assert cli._resolve_snapshot_date(ns) == date(2030, 1, 31)


def test_verify_one_requires_fixture_mode_in_v0_1_x():
    """v0.1.x has no real-data verify-one path; flag is now honored."""
    rc = cli.main(["verify-one", "--nct", "NCT01035255"])
    assert rc == 2  # deferred to v0.2.0
