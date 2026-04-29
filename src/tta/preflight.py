"""Preflight checks. Fail-closed before plan Task 2 ever runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests

from tta import config


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    action: str = ""


def _check_aact(aact_dir: Path) -> Check:
    if not aact_dir.is_dir():
        return Check("AACT snapshot directory", False, f"missing: {aact_dir}",
                    "Re-download AACT snapshot to the configured TTA_AACT_DIR")
    studies = aact_dir / "studies.txt"
    if not studies.is_file():
        return Check("AACT snapshot directory", False,
                    f"studies.txt not found in {aact_dir}",
                    "Re-extract AACT TSV snapshot")
    return Check("AACT snapshot directory", True, str(aact_dir))


def _check_pairwise70(pw_dir: Path, min_files: int = 590) -> Check:
    if not pw_dir.is_dir():
        return Check("Pairwise70 directory", False, f"missing: {pw_dir}",
                    "Restore Pairwise70 .rda files to the configured TTA_PAIRWISE70_DIR")
    rda_count = sum(1 for _ in pw_dir.glob("*.rda"))
    if rda_count < min_files:
        return Check("Pairwise70 directory", False,
                    f"found {rda_count} .rda files, expected >= {min_files}",
                    "Re-sync Pairwise70 dataset")
    return Check("Pairwise70 directory", True, f"{rda_count} .rda files")


def _check_dossiergap(fixture: Path) -> Check:
    if not fixture.is_file():
        return Check("DossierGap fixture", False, f"missing: {fixture}",
                    "Re-run DossierGap pipeline to produce v0.3.0 CSV")
    return Check("DossierGap fixture", True, str(fixture))


def _check_ollama(url: str, model: str) -> Check:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        names = {m["name"] for m in r.json().get("models", [])}
        if model not in names:
            return Check("Ollama model", False,
                        f"{model} not installed (have: {sorted(names)})",
                        f"Run: ollama pull {model}")
        return Check("Ollama model", True, f"{model} present at {url}")
    except Exception as e:
        return Check("Ollama service", False, f"unreachable: {e}",
                    f"Start ollama service or set TTA_OLLAMA_URL")


def _check_pubmed() -> Check:
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi",
            timeout=10,
        )
        r.raise_for_status()
        return Check("PubMed E-utilities", True, "reachable")
    except Exception as e:
        return Check("PubMed E-utilities", False, f"unreachable: {e}",
                    "Check network / NCBI status")


def run_checks(
    aact_dir: Optional[Path] = None,
    pairwise70_dir: Optional[Path] = None,
    dossiergap_fixture: Optional[Path] = None,
    check_ollama: bool = True,
    check_pubmed: bool = True,
) -> List[Check]:
    aact_dir = aact_dir or config.AACT_SNAPSHOT_DIR
    pairwise70_dir = pairwise70_dir or config.PAIRWISE70_DIR
    dossiergap_fixture = dossiergap_fixture or config.DOSSIERGAP_FIXTURE
    checks: List[Check] = [
        _check_aact(aact_dir),
        _check_pairwise70(pairwise70_dir),
        _check_dossiergap(dossiergap_fixture),
    ]
    if check_ollama:
        checks.append(_check_ollama(config.OLLAMA_URL, config.OLLAMA_MODEL))
    if check_pubmed:
        checks.append(_check_pubmed())
    return checks


def format_action_list(checks: Iterable[Check]) -> str:
    failed = [c for c in checks if not c.ok]
    if not failed:
        return "All preflight checks passed."
    lines = ["Preflight FAILURES (fix before continuing):"]
    for c in failed:
        lines.append(f"  - {c.name}: {c.detail}")
        if c.action:
            lines.append(f"      action: {c.action}")
    return "\n".join(lines)
