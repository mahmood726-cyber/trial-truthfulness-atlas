from __future__ import annotations

from pathlib import Path

import pytest

from tta import preflight


def test_preflight_all_present(tmp_path, monkeypatch):
    aact = tmp_path / "aact"
    aact.mkdir()
    (aact / "studies.txt").write_text("id|nct_id\n", encoding="utf-8")
    pw = tmp_path / "pw70"
    pw.mkdir()
    for i in range(595):
        (pw / f"CD{i:06d}.rda").write_bytes(b"")
    dg = tmp_path / "dg.csv"
    dg.write_text("nct_id\nNCT01\n", encoding="utf-8")

    checks = preflight.run_checks(
        aact_dir=aact,
        pairwise70_dir=pw,
        dossiergap_fixture=dg,
        check_ollama=False,
        check_pubmed=False,
    )
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_preflight_missing_aact(tmp_path):
    pw = tmp_path / "pw70"
    pw.mkdir()
    (pw / "x.rda").write_bytes(b"")
    dg = tmp_path / "dg.csv"
    dg.write_text("x\n", encoding="utf-8")

    checks = preflight.run_checks(
        aact_dir=tmp_path / "missing",
        pairwise70_dir=pw,
        dossiergap_fixture=dg,
        check_ollama=False,
        check_pubmed=False,
    )
    failures = [c for c in checks if not c.ok]
    assert any("AACT" in c.name for c in failures)


def test_preflight_too_few_pairwise70(tmp_path):
    aact = tmp_path / "aact"
    aact.mkdir()
    (aact / "studies.txt").write_text("id\n", encoding="utf-8")
    pw = tmp_path / "pw70"
    pw.mkdir()
    (pw / "only.rda").write_bytes(b"")
    dg = tmp_path / "dg.csv"
    dg.write_text("x\n", encoding="utf-8")

    checks = preflight.run_checks(
        aact_dir=aact,
        pairwise70_dir=pw,
        dossiergap_fixture=dg,
        check_ollama=False,
        check_pubmed=False,
    )
    failures = [c for c in checks if not c.ok]
    assert any("Pairwise70" in c.name for c in failures)


def test_format_action_list_lists_failures():
    from tta.preflight import Check, format_action_list

    failed = [Check(name="X", ok=False, detail="missing", action="install X")]
    out = format_action_list(failed)
    assert "install X" in out
    assert "missing" in out
