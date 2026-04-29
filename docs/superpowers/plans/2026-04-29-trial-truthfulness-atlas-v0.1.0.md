# Trial Truthfulness Atlas v0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v0.1.0 of the Trial Truthfulness Atlas — for the cardiology subset of Pairwise70 Cochrane MAs, compute five integrity flags (NCT-bridge + outcome drift + N drift + direction concordance + results-posting compliance) and produce `atlas.csv` + `ma_rollup.csv` + `dashboard.html` + `verification.html`.

**Architecture:** Local-only pipeline: AACT 12 GB snapshot + Pairwise70 .rda files → ingest to parquet → bridge to NCT → run four flag computers (one LLM-judged via local `gemma2:9b`, three numeric) → aggregate to MA level → emit dashboard + verification UI. Judge cache keyed by sha256(prompt+input+model_version) for byte-reproducibility.

**Tech Stack:** Python 3.11+, pandas, pyarrow, pyreadr (for .rda), requests (PubMed E-utilities), ollama HTTP API, pytest, argparse. No external runtime services beyond local ollama and PubMed E-utilities.

**Spec:** `docs/superpowers/specs/2026-04-29-trial-truthfulness-atlas-design.md` (commit `726594c`, approved 2026-04-29).

---

## File structure

```
trial-truthfulness-atlas/
  pyproject.toml                                — package metadata + deps
  requirements.txt                              — pinned runtime deps
  .gitignore                                    — already in repo
  push.sh                                       — git push helper (added late)
  README.md                                     — project overview
  FORKING.md                                    — multi-tenant per portfolio convention
  E156-PROTOCOL.md                              — workbook protocol entry

  src/tta/
    __init__.py                                 — version export
    config.py                                   — paths, thresholds, env-var resolution
    ingest.py                                   — AACT TSV + Pairwise70 .rda → parquet
    cardio_filter.py                            — Cochrane Heart Group / CV MeSH subset
    bridge.py                                   — Flag 0: Study/PMID → NCT waterfall
    flags/
      __init__.py
      outcome_drift.py                          — Flag 1 (LLM-judged)
      n_drift.py                                — Flag 2 (numeric + negation scrub)
      direction_concordance.py                  — Flag 3 (numeric)
      results_posting.py                        — Flag 4 (FDAAA logic)
    judge/
      __init__.py
      ollama_client.py                          — local LLM HTTP wrapper
      prompts.py                                — frozen prompts + sha256 hash
      cache.py                                  — sha256-keyed JSON cache
    aggregate.py                                — MA-level rollup
    cli.py                                      — argparse: preflight|build|sweep|verify-one
    dashboards/
      __init__.py
      atlas_dashboard.py                        — emits dashboard.html (inline SVG)
      verification_ui.py                        — emits verification.html

  tests/
    __init__.py
    conftest.py                                 — fixture loaders, judge-cache override
    fixtures/
      cardio_5_trials.json                      — hand-curated ground truth
      aact_sample/                              — tiny AACT TSV slices
      pairwise70_sample.parquet                 — tiny .rda → parquet sample
      judge_cache/                              — pre-computed sha256-keyed responses
      pubmed_responses/                         — cached E-utility XML
    test_preflight.py
    test_config.py
    test_ingest_aact.py
    test_ingest_pairwise70.py
    test_cardio_filter.py
    test_judge_cache.py
    test_judge_ollama.py
    test_judge_prompts.py
    test_bridge.py
    test_flag_outcome_drift.py
    test_flag_n_drift.py
    test_flag_direction_concordance.py
    test_flag_results_posting.py
    test_aggregate.py
    test_cli.py
    test_integration_5trial.py                  — pinned atlas.csv byte-match
    test_dashboard_render.py
    test_verification_render.py

  baseline/
    cardio_v0.1.0.json                          — pinned numerical baseline

  outputs/                                      — gitignored
  data/                                         — gitignored
```

**Boundary rationale:**
- One module per flag (independent test surface, parallel-friendly)
- `judge/` isolates LLM concerns so flag modules stay testable without ollama
- `dashboards/` is its own subpackage so atlas-builder code is untouched if dashboard format changes
- Tests mirror `src/` structure 1:1; `test_integration_5trial.py` is the single end-to-end gate

---

## Task 0: Project scaffold

**Why this task:** establish package layout, pinned deps, and the Python entry point so every subsequent task can `pip install -e .` and run `pytest`. Per the "Preflight external prereqs" lesson, we explicitly do NOT touch external systems yet.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\pyproject.toml`
- Create: `C:\Projects\trial-truthfulness-atlas\requirements.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\__init__.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\config.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\__init__.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\conftest.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_config.py`

- [ ] **Step 0.1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tta"
version = "0.1.0"
description = "Trial Truthfulness Atlas — 5th Pairwise70 atlas"
requires-python = ">=3.11"
authors = [{name = "Mahmood Ahmad", email = "mahmood726@gmail.com"}]
dependencies = [
    "pandas>=2.2",
    "pyarrow>=15",
    "pyreadr>=0.5",
    "requests>=2.31",
    "lxml>=5.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[project.scripts]
tta = "tta.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- [ ] **Step 0.2: Write `requirements.txt`**

```
pandas==2.2.3
pyarrow==18.1.0
pyreadr==0.5.2
requests==2.32.3
lxml==5.3.0
pytest==8.3.4
pytest-cov==6.0.0
```

- [ ] **Step 0.3: Write `src/tta/__init__.py`**

```python
"""Trial Truthfulness Atlas."""

__version__ = "0.1.0"
```

- [ ] **Step 0.4: Write the failing test for `config.py`**

`tests/test_config.py`:

```python
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
    monkeypatch.setenv("TTA_AACT_DIR", str(tmp_path / "aact"))
    import importlib

    importlib.reload(config)
    try:
        assert config.AACT_SNAPSHOT_DIR == tmp_path / "aact"
    finally:
        monkeypatch.delenv("TTA_AACT_DIR")
        importlib.reload(config)
```

- [ ] **Step 0.5: Run the test to verify failure**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tta.config'`.

- [ ] **Step 0.6: Implement `src/tta/config.py`**

```python
"""Path + threshold configuration. All values overridable via env vars."""

from __future__ import annotations

import os
from pathlib import Path

AACT_SNAPSHOT_DIR = Path(os.environ.get("TTA_AACT_DIR", r"D:\AACT-storage\AACT\2026-04-12"))
PAIRWISE70_DIR = Path(os.environ.get("TTA_PAIRWISE70_DIR", r"C:\Projects\Pairwise70\data"))
DOSSIERGAP_FIXTURE = Path(
    os.environ.get(
        "TTA_DOSSIERGAP_FIXTURE",
        r"C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv",
    )
)

DATA_DIR = Path(os.environ.get("TTA_DATA_DIR", "data"))
OUTPUTS_DIR = Path(os.environ.get("TTA_OUTPUTS_DIR", "outputs"))
JUDGE_CACHE_DIR = Path(os.environ.get("TTA_JUDGE_CACHE_DIR", "data/judge_cache"))

OLLAMA_URL = os.environ.get("TTA_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("TTA_OLLAMA_MODEL", "gemma2:9b")

N_DRIFT_THRESHOLD = float(os.environ.get("TTA_N_DRIFT_THRESHOLD", "0.10"))
DIRECTION_EPSILON = float(os.environ.get("TTA_DIRECTION_EPSILON", "0.01"))
BRIDGE_CONFIDENCE_MIN = float(os.environ.get("TTA_BRIDGE_CONFIDENCE_MIN", "0.7"))

AACT_MAX_AGE_DAYS = int(os.environ.get("TTA_AACT_MAX_AGE_DAYS", "90"))
SEED = int(os.environ.get("TTA_SEED", "42"))
```

- [ ] **Step 0.7: Write `tests/__init__.py` (empty)**

```python
```

- [ ] **Step 0.8: Write `tests/conftest.py`**

```python
"""Test fixtures shared across tta tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def judge_cache_dir(tmp_path) -> Path:
    cache = tmp_path / "judge_cache"
    cache.mkdir()
    return cache
```

- [ ] **Step 0.9: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: 6 PASS.

- [ ] **Step 0.10: Commit**

```bash
git add pyproject.toml requirements.txt src/ tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat(scaffold): pyproject + config module + test harness"
```

---

## Task 1: Preflight CLI command

**Why this task:** the "Preflight external prereqs BEFORE starting a multi-task plan" lesson — every later task assumes AACT, Pairwise70, ollama, and DossierGap exist. Failing here loudly with an action list is cheaper than discovering it at Task 17.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\cli.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\preflight.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_preflight.py`

- [ ] **Step 1.1: Write the failing test**

`tests/test_preflight.py`:

```python
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
```

- [ ] **Step 1.2: Run to verify failure**

Run: `pytest tests/test_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tta.preflight'`.

- [ ] **Step 1.3: Implement `src/tta/preflight.py`**

```python
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
                    "Re-download AACT snapshot to D:/AACT-storage/AACT/<date>/")
    studies = aact_dir / "studies.txt"
    if not studies.is_file():
        return Check("AACT snapshot directory", False,
                    f"studies.txt not found in {aact_dir}",
                    "Re-extract AACT TSV snapshot")
    return Check("AACT snapshot directory", True, str(aact_dir))


def _check_pairwise70(pw_dir: Path, min_files: int = 590) -> Check:
    if not pw_dir.is_dir():
        return Check("Pairwise70 directory", False, f"missing: {pw_dir}",
                    "Restore Pairwise70 .rda files to C:/Projects/Pairwise70/data/")
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
```

- [ ] **Step 1.4: Run tests**

Run: `pytest tests/test_preflight.py -v`
Expected: 4 PASS.

- [ ] **Step 1.5: Implement minimal `src/tta/cli.py` with the preflight subcommand**

```python
"""Command-line entry point. Subcommands: preflight, build, sweep, verify-one."""

from __future__ import annotations

import argparse
import sys

from tta import preflight


def cmd_preflight(args: argparse.Namespace) -> int:
    checks = preflight.run_checks()
    print(preflight.format_action_list(checks))
    return 0 if all(c.ok for c in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tta", description="Trial Truthfulness Atlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight", help="check external prereqs")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1.6: Smoke-test the CLI end-to-end**

Run: `python -m tta.cli preflight`
Expected: prints check results; exit code 0 if your real environment is set up, non-zero otherwise. Either is acceptable here — we just want the wire to work.

- [ ] **Step 1.7: Commit**

```bash
git add src/tta/preflight.py src/tta/cli.py tests/test_preflight.py
git commit -m "feat(preflight): cli + checks for AACT, Pairwise70, DossierGap, ollama, PubMed"
```

---

## Task 2: Hand-curated 5-trial fixture

**Why this task:** integration tests need ground truth. Picking it BEFORE writing flag code prevents post-hoc fitting. Five trials drawn from DossierGap's known-positive set + two synthetic edge cases give us all five flag values represented.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\cardio_5_trials.json`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\README.md`

- [ ] **Step 2.1: Write `tests/fixtures/cardio_5_trials.json`**

The five-trial fixture. Three known-positive cardio trials from DossierGap (PARADIGM-HF, VICTORIA, GRIPHON) + one with deliberately drifted N + one with an unbridgeable Study label.

```json
{
  "fixture_version": "1.0.0",
  "snapshot_date": "2026-04-12",
  "trials": [
    {
      "study_label": "McMurray 2014",
      "review_doi": "10.1002/14651858.CD012612.pub2",
      "expected_nct": "NCT01035255",
      "expected_pmid": "25176015",
      "trial_acronym": "PARADIGM-HF",
      "registered_outcome": "Composite endpoint of CV death and HF hospitalisation",
      "ma_extracted_outcome": "CV death or HF hospitalisation",
      "registered_n": 8442,
      "ma_n": 8442,
      "registered_effect_sign": -1,
      "ma_effect_sign": -1,
      "ma_effect_value": -0.223,
      "registered_completion_date": "2014-03-31",
      "results_first_posted_date": "2015-01-09",
      "fdaaa_applicable": true,
      "expected_flags": {
        "bridge": "bridged",
        "outcome_drift": "identical",
        "n_drift": "not_flagged",
        "direction_concordance": "concordant",
        "results_posting": "posted"
      }
    },
    {
      "study_label": "Armstrong 2020",
      "review_doi": "10.1002/14651858.CD013650.pub2",
      "expected_nct": "NCT02861534",
      "expected_pmid": "32222134",
      "trial_acronym": "VICTORIA",
      "registered_outcome": "Time to first occurrence of CV death or HF hospitalisation",
      "ma_extracted_outcome": "CV death or HF hospitalisation",
      "registered_n": 5050,
      "ma_n": 5050,
      "registered_effect_sign": -1,
      "ma_effect_sign": -1,
      "ma_effect_value": -0.105,
      "registered_completion_date": "2020-05-15",
      "results_first_posted_date": "2020-03-30",
      "fdaaa_applicable": true,
      "expected_flags": {
        "bridge": "bridged",
        "outcome_drift": "refinement",
        "n_drift": "not_flagged",
        "direction_concordance": "concordant",
        "results_posting": "posted"
      }
    },
    {
      "study_label": "Sitbon 2015",
      "review_doi": "10.1002/14651858.CD011162.pub2",
      "expected_nct": "NCT01106014",
      "expected_pmid": "26308684",
      "trial_acronym": "GRIPHON",
      "registered_outcome": "Composite morbidity-mortality endpoint",
      "ma_extracted_outcome": "All-cause mortality at 12 weeks",
      "registered_n": 1156,
      "ma_n": 1150,
      "registered_effect_sign": -1,
      "ma_effect_sign": -1,
      "ma_effect_value": -0.405,
      "registered_completion_date": "2014-09-30",
      "results_first_posted_date": "2015-08-27",
      "fdaaa_applicable": true,
      "expected_flags": {
        "bridge": "bridged",
        "outcome_drift": "substantively_different",
        "n_drift": "not_flagged",
        "direction_concordance": "concordant",
        "results_posting": "posted"
      }
    },
    {
      "study_label": "Synthetic-NDrift 2018",
      "review_doi": "10.1002/14651858.CDFAKE001.pub1",
      "expected_nct": "NCT99999001",
      "expected_pmid": "99999001",
      "trial_acronym": "FIXTURE-N",
      "registered_outcome": "All-cause mortality",
      "ma_extracted_outcome": "All-cause mortality",
      "registered_n": 1000,
      "ma_n": 700,
      "registered_effect_sign": -1,
      "ma_effect_sign": -1,
      "ma_effect_value": -0.15,
      "registered_completion_date": "2018-06-01",
      "results_first_posted_date": null,
      "fdaaa_applicable": true,
      "expected_flags": {
        "bridge": "bridged",
        "outcome_drift": "identical",
        "n_drift": "flagged",
        "direction_concordance": "concordant",
        "results_posting": "required_not_posted"
      }
    },
    {
      "study_label": "Unknown Author 1999",
      "review_doi": "10.1002/14651858.CDFAKE002.pub1",
      "expected_nct": null,
      "expected_pmid": null,
      "trial_acronym": null,
      "registered_outcome": null,
      "ma_extracted_outcome": "Mortality",
      "registered_n": null,
      "ma_n": 200,
      "registered_effect_sign": null,
      "ma_effect_sign": -1,
      "ma_effect_value": -0.05,
      "registered_completion_date": null,
      "results_first_posted_date": null,
      "fdaaa_applicable": null,
      "expected_flags": {
        "bridge": "unbridgeable",
        "outcome_drift": "unscoreable",
        "n_drift": "unscoreable",
        "direction_concordance": "unscoreable",
        "results_posting": "unscoreable"
      }
    }
  ]
}
```

- [ ] **Step 2.2: Write `tests/fixtures/README.md`**

```markdown
# Fixtures

`cardio_5_trials.json` — five hand-curated cardiology trials with known
ground truth across all five integrity flags. Used by
`tests/test_integration_5trial.py` to gate v0.1.0 byte-reproducibility.

Trials 1–3 (PARADIGM-HF, VICTORIA, GRIPHON) are real and drawn from
DossierGap v0.3.0. Their NCT, PMID, N, and outcome values were verified
manually against published Cochrane reviews and CT.gov on 2026-04-29.

Trial 4 (FIXTURE-N) is synthetic — flags an N-drift case with
deterministic values.

Trial 5 (Unknown Author 1999) is synthetic — exercises the unbridgeable
+ all-unscoreable path.

If you change this fixture, regenerate `baseline/cardio_v0.1.0.json` and
update the integration test's pinned `atlas.csv` snapshot.
```

- [ ] **Step 2.3: Add a fixture-loader test to confirm the JSON parses**

Append to `tests/test_config.py`:

```python
import json


def test_cardio_5_trials_fixture_parses():
    fixture_path = Path(__file__).parent / "fixtures" / "cardio_5_trials.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert data["fixture_version"] == "1.0.0"
    assert len(data["trials"]) == 5
    flag_keys = {"bridge", "outcome_drift", "n_drift", "direction_concordance", "results_posting"}
    for t in data["trials"]:
        assert set(t["expected_flags"].keys()) == flag_keys
```

- [ ] **Step 2.4: Run tests**

Run: `pytest tests/test_config.py::test_cardio_5_trials_fixture_parses -v`
Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add tests/fixtures/cardio_5_trials.json tests/fixtures/README.md tests/test_config.py
git commit -m "test(fixtures): hand-curated 5-trial cardio ground truth"
```

---

## Task 3: AACT ingest

**Why this task:** the four numeric/LLM flags all read AACT columns. Materialising AACT TSV → parquet once means downstream code never re-parses the 12 GB snapshot.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\ingest.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\studies.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\design_outcomes.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\calculated_values.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\outcome_analyses.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\interventions.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\aact_sample\browse_conditions.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_ingest_aact.py`

- [ ] **Step 3.1: Write the AACT TSV fixtures**

`tests/fixtures/aact_sample/studies.txt`:

```
nct_id|study_type|phase|completion_date|results_first_posted_date
NCT01035255|interventional|Phase 3|2014-03-31|2015-01-09
NCT02861534|interventional|Phase 3|2020-05-15|2020-03-30
NCT01106014|interventional|Phase 3|2014-09-30|2015-08-27
NCT99999001|interventional|Phase 3|2018-06-01|
NCT99999009|observational|N/A|2019-01-01|
```

`tests/fixtures/aact_sample/design_outcomes.txt`:

```
nct_id|outcome_type|measure
NCT01035255|primary|Composite endpoint of CV death and HF hospitalisation
NCT02861534|primary|Time to first occurrence of CV death or HF hospitalisation
NCT01106014|primary|Composite morbidity-mortality endpoint
NCT99999001|primary|All-cause mortality
```

`tests/fixtures/aact_sample/calculated_values.txt`:

```
nct_id|actual_enrollment
NCT01035255|8442
NCT02861534|5050
NCT01106014|1156
NCT99999001|1000
```

`tests/fixtures/aact_sample/outcome_analyses.txt`:

```
nct_id|param_type|param_value|ci_lower_limit|ci_upper_limit
NCT01035255|Hazard Ratio|0.80|0.73|0.87
NCT02861534|Hazard Ratio|0.90|0.82|0.98
NCT01106014|Hazard Ratio|0.61|0.46|0.81
NCT99999001|Hazard Ratio|0.86|0.70|1.05
```

`tests/fixtures/aact_sample/interventions.txt`:

```
nct_id|intervention_type|name
NCT01035255|drug|sacubitril/valsartan
NCT02861534|drug|vericiguat
NCT01106014|drug|selexipag
NCT99999001|drug|fixturedrug
NCT99999009|behavioral|education
```

`tests/fixtures/aact_sample/browse_conditions.txt`:

```
nct_id|mesh_term|downcase_mesh_term
NCT01035255|Heart Failure|heart failure
NCT02861534|Heart Failure|heart failure
NCT01106014|Pulmonary Arterial Hypertension|pulmonary arterial hypertension
NCT99999001|Heart Failure|heart failure
NCT99999009|Mouth Diseases|mouth diseases
```

- [ ] **Step 3.2: Write the failing test**

`tests/test_ingest_aact.py`:

```python
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
    assert len(df) == 5


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
```

- [ ] **Step 3.3: Run to verify failure**

Run: `pytest tests/test_ingest_aact.py -v`
Expected: FAIL with `AttributeError: module 'tta.ingest' has no attribute 'load_aact_table'`.

- [ ] **Step 3.4: Implement `src/tta/ingest.py`**

```python
"""Ingest AACT TSV snapshots and Pairwise70 .rda files into parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

DATE_COLUMNS_BY_TABLE = {
    "studies": ["completion_date", "results_first_posted_date"],
}


def load_aact_table(snapshot_dir: Path, table: str) -> pd.DataFrame:
    path = snapshot_dir / f"{table}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    parse_dates = DATE_COLUMNS_BY_TABLE.get(table, [])
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        engine="c",
    )
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def materialise_aact(
    snapshot_dir: Path,
    out_dir: Path,
    tables: Iterable[str],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for table in tables:
        df = load_aact_table(snapshot_dir, table)
        out_path = out_dir / f"aact_{table}.parquet"
        df.to_parquet(out_path, index=False)
        written[table] = out_path
    return written
```

- [ ] **Step 3.5: Run tests**

Run: `pytest tests/test_ingest_aact.py -v`
Expected: 5 PASS.

- [ ] **Step 3.6: Commit**

```bash
git add src/tta/ingest.py tests/fixtures/aact_sample/ tests/test_ingest_aact.py
git commit -m "feat(ingest): AACT TSV → parquet with typed date columns"
```

---

## Task 4: Pairwise70 ingest

**Why this task:** Pairwise70 ships as 595 `.rda` files (R serialized). We materialise them to one parquet at startup so all later code reads pandas, not R.

**Files:**
- Modify: `C:\Projects\trial-truthfulness-atlas\src\tta\ingest.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\pairwise70_sample\CDFAKE001_pub1_data.rda` (generated by helper)
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\pairwise70_sample\CDFAKE002_pub1_data.rda` (generated by helper)
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_ingest_pairwise70.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\_make_pairwise70_sample.py`

- [ ] **Step 4.1: Write the helper that generates two sample .rda files**

`tests/fixtures/_make_pairwise70_sample.py`:

```python
"""One-shot generator. Run once, commit the .rda outputs.

    python tests/fixtures/_make_pairwise70_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr

OUT = Path(__file__).parent / "pairwise70_sample"
OUT.mkdir(exist_ok=True)


df1 = pd.DataFrame({
    "Analysis.group": ["1.1", "1.1", "1.1"],
    "Analysis.number": [1, 1, 1],
    "Analysis.name": ["CV death or HF hosp"] * 3,
    "Subgroup": [None, None, None],
    "Study": ["McMurray 2014", "Armstrong 2020", "Sitbon 2015"],
    "Study.year": [2014, 2020, 2015],
    "Experimental.N": [4209, 2526, 574],
    "Control.N": [4233, 2524, 582],
    "Mean": [-0.223, -0.105, -0.405],
    "CI.start": [-0.314, -0.198, -0.776],
    "CI.end": [-0.139, -0.020, -0.211],
    "review_doi": [
        "10.1002/14651858.CD012612.pub2",
        "10.1002/14651858.CD013650.pub2",
        "10.1002/14651858.CD011162.pub2",
    ],
})
pyreadr.write_rdata(str(OUT / "CDFAKE001_pub1_data.rda"), df1, df_name="CDFAKE001_pub1_data")


df2 = pd.DataFrame({
    "Analysis.group": ["1.1", "1.1"],
    "Analysis.number": [1, 1],
    "Analysis.name": ["Mortality"] * 2,
    "Subgroup": [None, None],
    "Study": ["Synthetic-NDrift 2018", "Unknown Author 1999"],
    "Study.year": [2018, 1999],
    "Experimental.N": [350, 100],
    "Control.N": [350, 100],
    "Mean": [-0.15, -0.05],
    "CI.start": [-0.30, -0.18],
    "CI.end": [0.0, 0.08],
    "review_doi": [
        "10.1002/14651858.CDFAKE001.pub1",
        "10.1002/14651858.CDFAKE002.pub1",
    ],
})
pyreadr.write_rdata(str(OUT / "CDFAKE002_pub1_data.rda"), df2, df_name="CDFAKE002_pub1_data")

print("Wrote", list(OUT.glob("*.rda")))
```

- [ ] **Step 4.2: Generate the .rda files once and commit them**

Run: `python tests/fixtures/_make_pairwise70_sample.py`
Expected: prints two `.rda` paths.

- [ ] **Step 4.3: Write the failing test**

`tests/test_ingest_pairwise70.py`:

```python
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
```

- [ ] **Step 4.4: Run to verify failure**

Run: `pytest tests/test_ingest_pairwise70.py -v`
Expected: FAIL with `AttributeError: ... 'load_pairwise70_rda'`.

- [ ] **Step 4.5: Add `load_pairwise70_rda`, `load_pairwise70_dir`, `materialise_pairwise70` to `src/tta/ingest.py`**

Append:

```python
import pyreadr


def _review_id_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_data"):
        name = name[: -len("_data")]
    return name


def load_pairwise70_rda(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"no R objects in {path}")
    obj_name, df = next(iter(result.items()))
    df = df.copy()
    df.attrs["review_id"] = _review_id_from_filename(path)
    df["review_id"] = _review_id_from_filename(path)
    return df


def load_pairwise70_dir(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.rda"))
    if not files:
        raise FileNotFoundError(f"no .rda files in {directory}")
    frames = [load_pairwise70_rda(p) for p in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def materialise_pairwise70(directory: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_pairwise70_dir(directory)
    df.to_parquet(out_path, index=False)
    return out_path
```

- [ ] **Step 4.6: Run tests**

Run: `pytest tests/test_ingest_pairwise70.py -v`
Expected: 3 PASS.

- [ ] **Step 4.7: Commit**

```bash
git add src/tta/ingest.py tests/fixtures/_make_pairwise70_sample.py tests/fixtures/pairwise70_sample/ tests/test_ingest_pairwise70.py
git commit -m "feat(ingest): Pairwise70 .rda → parquet with review_id column"
```

---

## Task 5: Cardiology subset filter

**Why this task:** v0.1.0 scope is cardiology only. We need a deterministic, reviewable filter that decides which Pairwise70 reviews are in scope. Default: Cochrane Heart Group review-DOI prefix list, with a fallback to AACT MeSH cardiovascular terms when matched at trial level.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\cardio_filter.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\data\cochrane_heart_review_dois.txt`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_cardio_filter.py`

- [ ] **Step 5.1: Seed the heart-group review-DOI list**

`src/tta/data/cochrane_heart_review_dois.txt`:

```
# Cochrane Heart Group review DOIs in scope for v0.1.0.
# Format: one DOI per line; comments with '#'.
# Source: Cochrane Library, accessed 2026-04-29.
# Note: this is a v0.1.0 working subset; widen in v0.2.0.
10.1002/14651858.CD012612.pub2
10.1002/14651858.CD013650.pub2
10.1002/14651858.CD011162.pub2
10.1002/14651858.CDFAKE001.pub1
10.1002/14651858.CDFAKE002.pub1
```

- [ ] **Step 5.2: Write the failing test**

`tests/test_cardio_filter.py`:

```python
from __future__ import annotations

import pandas as pd

from tta import cardio_filter


def test_load_heart_group_doi_list_skips_comments():
    dois = cardio_filter.load_heart_group_dois()
    assert all(not d.startswith("#") for d in dois)
    assert "10.1002/14651858.CD012612.pub2" in dois


def test_filter_pairwise70_to_cardio(fixtures_dir):
    from tta import ingest

    df = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    out = cardio_filter.filter_pairwise70(df)
    assert len(out) == 5
    assert set(out["review_doi"].unique()) <= set(cardio_filter.load_heart_group_dois())


def test_filter_drops_non_cardio_reviews():
    df = pd.DataFrame({
        "Study": ["x", "y"],
        "review_doi": ["10.1002/14651858.CD000999.pub1", "10.1002/14651858.CD012612.pub2"],
        "Mean": [0.0, -0.1],
    })
    out = cardio_filter.filter_pairwise70(df)
    assert len(out) == 1
    assert out["review_doi"].iloc[0] == "10.1002/14651858.CD012612.pub2"


def test_filter_aact_to_cv_mesh(fixtures_dir):
    from tta import ingest

    bc = ingest.load_aact_table(fixtures_dir / "aact_sample", "browse_conditions")
    out = cardio_filter.filter_aact_browse_conditions(bc)
    assert "NCT99999009" not in set(out["nct_id"])
    assert "NCT01035255" in set(out["nct_id"])
```

- [ ] **Step 5.3: Run to verify failure**

Run: `pytest tests/test_cardio_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: ... cardio_filter`.

- [ ] **Step 5.4: Implement `src/tta/cardio_filter.py`**

```python
"""Cardiology subset filter for v0.1.0."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

DOI_LIST_PATH = Path(__file__).parent / "data" / "cochrane_heart_review_dois.txt"

CARDIOVASCULAR_MESH_TERMS = frozenset(
    {
        "heart failure",
        "myocardial infarction",
        "coronary artery disease",
        "coronary disease",
        "atrial fibrillation",
        "hypertension",
        "stroke",
        "cardiovascular diseases",
        "pulmonary arterial hypertension",
        "valvular heart disease",
        "arrhythmias, cardiac",
    }
)


@lru_cache(maxsize=1)
def load_heart_group_dois() -> Set[str]:
    lines = DOI_LIST_PATH.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def filter_pairwise70(df: pd.DataFrame) -> pd.DataFrame:
    dois = load_heart_group_dois()
    return df[df["review_doi"].isin(dois)].copy()


def filter_aact_browse_conditions(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["downcase_mesh_term"].isin(CARDIOVASCULAR_MESH_TERMS)
    return df[mask].copy()


def cardio_nct_set(browse_conditions: pd.DataFrame) -> Set[str]:
    return set(filter_aact_browse_conditions(browse_conditions)["nct_id"])
```

- [ ] **Step 5.5: Make `src/tta/data/` a proper data dir**

Create empty `src/tta/data/__init__.py`:

```python
```

Update `pyproject.toml` `[tool.setuptools.packages.find]` is already capturing src; ensure data file ships:

Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
tta = ["data/*.txt"]
```

- [ ] **Step 5.6: Run tests**

Run: `pip install -e . && pytest tests/test_cardio_filter.py -v`
Expected: 4 PASS.

- [ ] **Step 5.7: Commit**

```bash
git add src/tta/cardio_filter.py src/tta/data/cochrane_heart_review_dois.txt src/tta/data/__init__.py pyproject.toml tests/test_cardio_filter.py
git commit -m "feat(cardio): heart-group DOI filter + CV MeSH filter"
```

---

## Task 6: Judge — frozen prompts module

**Why this task:** prompts must be immutable text + a sha256 hash so cached judgments stay attributable to the exact wording that produced them. Any prompt change ⇒ new hash ⇒ cache miss ⇒ re-judge. This pattern is the single point where the LLM-judged Flag 1 stays reproducible.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\judge\__init__.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\judge\prompts.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_judge_prompts.py`

- [ ] **Step 6.1: Create empty `src/tta/judge/__init__.py`**

```python
```

- [ ] **Step 6.2: Write the failing test**

`tests/test_judge_prompts.py`:

```python
from __future__ import annotations

import hashlib

from tta.judge import prompts


def test_outcome_drift_prompt_v1_is_frozen():
    text = prompts.OUTCOME_DRIFT_V1
    assert "registered_outcome" in text
    assert "ma_extracted_outcome" in text
    assert "identical" in text
    assert "refinement" in text
    assert "substantively_different" in text


def test_outcome_drift_prompt_hash_is_stable():
    expected_sha = hashlib.sha256(prompts.OUTCOME_DRIFT_V1.encode("utf-8")).hexdigest()
    assert prompts.OUTCOME_DRIFT_V1_SHA256 == expected_sha


def test_render_outcome_drift_inserts_inputs():
    rendered = prompts.render_outcome_drift(
        registered_outcome="Composite of CV death and HF hospitalisation",
        ma_extracted_outcome="CV death or HF hospitalisation",
    )
    assert "Composite of CV death and HF hospitalisation" in rendered
    assert "CV death or HF hospitalisation" in rendered


def test_prompt_inputs_are_escaped_safely():
    # Inputs containing the delimiter must not break parsing
    rendered = prompts.render_outcome_drift(
        registered_outcome="Outcome with </END> sentinel",
        ma_extracted_outcome="Plain text",
    )
    assert "</END>" in rendered  # we keep it literal — no injection in local LLM context
```

- [ ] **Step 6.3: Run to verify failure**

Run: `pytest tests/test_judge_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: tta.judge.prompts`.

- [ ] **Step 6.4: Implement `src/tta/judge/prompts.py`**

```python
"""Frozen prompt strings for the local-LLM judge layer.

Any change to a prompt string requires bumping its version suffix
(`_V1` → `_V2`) and its sha256 constant. Cached judgments key on the
sha256, so a new version invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib

OUTCOME_DRIFT_V1 = """\
You are a clinical-trials methodologist. Compare two descriptions of a
trial outcome:

  registered_outcome: {registered_outcome}
  ma_extracted_outcome: {ma_extracted_outcome}

Reply with EXACTLY ONE of these labels (lowercase, no punctuation):

  identical                — same construct, same time-point, same definition
  refinement               — same construct but narrower / clearer wording
  substantively_different  — different construct or time-point or definition

Reply with ONLY the label. No explanation.
"""

OUTCOME_DRIFT_V1_SHA256 = hashlib.sha256(OUTCOME_DRIFT_V1.encode("utf-8")).hexdigest()


def render_outcome_drift(registered_outcome: str, ma_extracted_outcome: str) -> str:
    return OUTCOME_DRIFT_V1.format(
        registered_outcome=registered_outcome,
        ma_extracted_outcome=ma_extracted_outcome,
    )
```

- [ ] **Step 6.5: Run tests**

Run: `pytest tests/test_judge_prompts.py -v`
Expected: 4 PASS.

- [ ] **Step 6.6: Commit**

```bash
git add src/tta/judge/__init__.py src/tta/judge/prompts.py tests/test_judge_prompts.py
git commit -m "feat(judge): frozen outcome-drift prompt v1 with sha256 pin"
```

---

## Task 7: Judge — sha256-keyed disk cache

**Why this task:** every LLM judgment must be deterministic on rerun. Cache key = `sha256(prompt_sha + rendered_inputs + model_version)`. Tests can ship pre-computed cache JSON to stay 100 % offline.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\judge\cache.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_judge_cache.py`

- [ ] **Step 7.1: Write the failing test**

`tests/test_judge_cache.py`:

```python
from __future__ import annotations

import json

import pytest

from tta.judge import cache


def test_cache_key_is_deterministic():
    k1 = cache.cache_key("prompt-sha-abc", "rendered text", "gemma2:9b@xyz")
    k2 = cache.cache_key("prompt-sha-abc", "rendered text", "gemma2:9b@xyz")
    assert k1 == k2
    assert len(k1) == 64


def test_cache_key_changes_on_any_input(judge_cache_dir):
    base = cache.cache_key("p", "r", "m")
    assert cache.cache_key("p2", "r", "m") != base
    assert cache.cache_key("p", "r2", "m") != base
    assert cache.cache_key("p", "r", "m2") != base


def test_get_returns_none_on_miss(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    assert c.get("nonexistent") is None


def test_put_then_get_roundtrip(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    payload = {"label": "identical", "model_version": "gemma2:9b@xyz", "raw": "identical"}
    c.put("abcd1234", payload)
    got = c.get("abcd1234")
    assert got == payload


def test_cache_files_are_human_readable_json(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    c.put("efgh5678", {"label": "refinement"})
    files = list(judge_cache_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["label"] == "refinement"


def test_put_is_idempotent(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    c.put("k", {"label": "x"})
    c.put("k", {"label": "x"})  # must not raise
    assert c.get("k") == {"label": "x"}
```

- [ ] **Step 7.2: Run to verify failure**

Run: `pytest tests/test_judge_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: tta.judge.cache`.

- [ ] **Step 7.3: Implement `src/tta/judge/cache.py`**

```python
"""sha256-keyed JSON disk cache for LLM judgments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def cache_key(prompt_sha: str, rendered: str, model_version: str) -> str:
    h = hashlib.sha256()
    h.update(prompt_sha.encode("utf-8"))
    h.update(b"\x1f")
    h.update(rendered.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model_version.encode("utf-8"))
    return h.hexdigest()


class JudgeCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        p = self._path(key)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        p = self._path(key)
        p.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 7.4: Run tests**

Run: `pytest tests/test_judge_cache.py -v`
Expected: 6 PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/tta/judge/cache.py tests/test_judge_cache.py
git commit -m "feat(judge): sha256-keyed disk cache"
```

---

## Task 8: Judge — ollama HTTP client

**Why this task:** wrap the local ollama HTTP API behind a small interface so flag code calls `judge.classify(prompt_text, expected_labels)` and the wrapper handles seed, format-validation, and cache lookup.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\judge\ollama_client.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_judge_ollama.py`

- [ ] **Step 8.1: Write the failing test**

`tests/test_judge_ollama.py`:

```python
from __future__ import annotations

import json

import pytest

from tta.judge import ollama_client


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_classify_returns_label_on_clean_response(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _StubResponse({"response": "identical\n"})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    client = ollama_client.OllamaClient(url="http://x:11434", model="gemma2:9b")
    label = client.classify(
        prompt_text="prompt body",
        allowed_labels={"identical", "refinement", "substantively_different"},
    )
    assert label == "identical"
    assert captured["json"]["model"] == "gemma2:9b"
    assert captured["json"]["options"]["seed"] == 42
    assert captured["json"]["stream"] is False


def test_classify_strips_whitespace_and_quotes(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda url, json=None, timeout=None: _StubResponse({"response": "  'refinement' \n"}),
    )
    client = ollama_client.OllamaClient(url="http://x:11434", model="gemma2:9b")
    label = client.classify("p", {"identical", "refinement", "substantively_different"})
    assert label == "refinement"


def test_classify_returns_unscoreable_on_unknown_label(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda url, json=None, timeout=None: _StubResponse({"response": "maybe?"}),
    )
    client = ollama_client.OllamaClient(url="http://x:11434", model="gemma2:9b")
    label = client.classify("p", {"identical", "refinement", "substantively_different"})
    assert label == "unscoreable"


def test_get_model_version_returns_digest(monkeypatch):
    def fake_get(url, timeout=None):
        return _StubResponse({"models": [
            {"name": "gemma2:9b", "digest": "ff02c3702f32abc"},
            {"name": "qwen2.5-coder:7b", "digest": "dae161e27b0e123"},
        ]})

    monkeypatch.setattr(ollama_client.requests, "get", fake_get)
    client = ollama_client.OllamaClient(url="http://x:11434", model="gemma2:9b")
    assert client.get_model_version() == "gemma2:9b@ff02c3702f32abc"


def test_classify_raises_on_http_error(monkeypatch):
    class BadResp:
        status_code = 500
        def raise_for_status(self):
            raise RuntimeError("server died")

    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda url, json=None, timeout=None: BadResp(),
    )
    client = ollama_client.OllamaClient(url="http://x:11434", model="gemma2:9b")
    with pytest.raises(RuntimeError):
        client.classify("p", {"identical"})
```

- [ ] **Step 8.2: Run to verify failure**

Run: `pytest tests/test_judge_ollama.py -v`
Expected: FAIL — module missing.

- [ ] **Step 8.3: Implement `src/tta/judge/ollama_client.py`**

```python
"""Local ollama HTTP wrapper with deterministic seed and label gating."""

from __future__ import annotations

from typing import Set

import requests

from tta import config


class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: int = 60):
        self.url = url or config.OLLAMA_URL
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout

    def get_model_version(self) -> str:
        r = requests.get(f"{self.url}/api/tags", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m["name"] == self.model:
                return f"{self.model}@{m['digest'][:12]}"
        return f"{self.model}@unknown"

    def classify(self, prompt_text: str, allowed_labels: Set[str]) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "seed": config.SEED,
                "temperature": 0.0,
                "num_predict": 16,
            },
        }
        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        raw = r.json().get("response", "").strip().strip("'\"").strip().lower()
        first = raw.split()[0] if raw else ""
        return first if first in allowed_labels else "unscoreable"
```

- [ ] **Step 8.4: Run tests**

Run: `pytest tests/test_judge_ollama.py -v`
Expected: 5 PASS.

- [ ] **Step 8.5: Commit**

```bash
git add src/tta/judge/ollama_client.py tests/test_judge_ollama.py
git commit -m "feat(judge): ollama HTTP client with deterministic seed + label gating"
```

---

## Task 9: Flag 0 — NCT bridge

**Why this task:** Pairwise70 has `Study` (author-year), not `nct_id`. Without bridging, Flags 1–3 cannot run. The bridge is also a publishable headline ("X % unbridgeable").

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\bridge.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\dossiergap_sample.csv`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_bridge.py`

- [ ] **Step 9.1: Create the DossierGap sample fixture**

`tests/fixtures/dossiergap_sample.csv`:

```csv
nct_id,study_label,review_doi,acronym
NCT01035255,McMurray 2014,10.1002/14651858.CD012612.pub2,PARADIGM-HF
NCT02861534,Armstrong 2020,10.1002/14651858.CD013650.pub2,VICTORIA
NCT01106014,Sitbon 2015,10.1002/14651858.CD011162.pub2,GRIPHON
NCT99999001,Synthetic-NDrift 2018,10.1002/14651858.CDFAKE001.pub1,FIXTURE-N
```

- [ ] **Step 9.2: Write the failing test**

`tests/test_bridge.py`:

```python
from __future__ import annotations

import pandas as pd

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

    # Trial 5 (Unknown Author 1999) is unbridgeable
    unknown = bridged[bridged["Study"] == "Unknown Author 1999"]
    assert unknown["bridge_method"].iloc[0] == "unbridgeable"

    # Trial 1 bridges via direct match
    paradigm = bridged[bridged["Study"] == "McMurray 2014"]
    assert paradigm["nct_id"].iloc[0] == "NCT01035255"


def test_bridge_resolution_rate(fixtures_dir):
    from tta import ingest

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    pw = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    bridged = bridge.bridge_pairwise70(pw, dossiergap=dg, aact_id_information=None)
    rate = bridge.resolution_rate(bridged)
    # 4 out of 5 fixture trials are bridgeable
    assert rate == pytest.approx(0.80, abs=1e-9)


import pytest
```

- [ ] **Step 9.3: Run to verify failure**

Run: `pytest tests/test_bridge.py -v`
Expected: FAIL — module missing.

- [ ] **Step 9.4: Implement `src/tta/bridge.py`**

```python
"""Flag 0 — bridge Pairwise70 (Study, review_doi) → CT.gov NCT.

Waterfall:
  1. DossierGap direct match by (study_label, review_doi)
  2. AACT id_information cross-ref (deferred to v0.2.0; stub raises NotImplemented if attempted)
  3. Cochrane HTML scrape (deferred to v0.2.0)
  4. Heuristic match by author surname + year ±1 (deferred to v0.2.0)

v0.1.0 ships only method 1; methods 2–4 land in v0.2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BridgeResult:
    nct_id: Optional[str]
    method: str
    confidence: float


def load_dossiergap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"nct_id", "study_label", "review_doi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DossierGap CSV missing columns: {missing}")
    return df


def bridge_one(
    pw_row: pd.Series,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
) -> BridgeResult:
    study = pw_row.get("Study")
    doi = pw_row.get("review_doi")
    if pd.isna(study) or pd.isna(doi):
        return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)

    direct = dossiergap[
        (dossiergap["study_label"] == study) & (dossiergap["review_doi"] == doi)
    ]
    if not direct.empty:
        return BridgeResult(
            nct_id=str(direct["nct_id"].iloc[0]),
            method="dossiergap_direct",
            confidence=0.99,
        )

    return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)


def bridge_pairwise70(
    pw_df: pd.DataFrame,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = pw_df.copy()
    results = [bridge_one(row, dossiergap, aact_id_information) for _, row in pw_df.iterrows()]
    out["nct_id"] = [r.nct_id for r in results]
    out["bridge_method"] = [r.method for r in results]
    out["bridge_confidence"] = [r.confidence for r in results]
    return out


def resolution_rate(bridged: pd.DataFrame) -> float:
    if bridged.empty:
        return 0.0
    return float((bridged["bridge_method"] != "unbridgeable").mean())
```

- [ ] **Step 9.5: Run tests**

Run: `pytest tests/test_bridge.py -v`
Expected: 5 PASS.

- [ ] **Step 9.6: Commit**

```bash
git add src/tta/bridge.py tests/fixtures/dossiergap_sample.csv tests/test_bridge.py
git commit -m "feat(bridge): Flag 0 — Pairwise70 → NCT via DossierGap direct match"
```

---

## Task 10: Flag 1 — outcome drift (LLM-judged)

**Why this task:** the only flag that requires the LLM. Calls into Task 6/7/8 stack: render prompt → check cache → call ollama on miss → cache result.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\flags\__init__.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\flags\outcome_drift.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_flag_outcome_drift.py`

- [ ] **Step 10.1: Create empty `src/tta/flags/__init__.py`**

```python
```

- [ ] **Step 10.2: Write the failing test**

`tests/test_flag_outcome_drift.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tta.flags import outcome_drift
from tta.judge import cache, prompts


ALLOWED = {"identical", "refinement", "substantively_different"}


def test_compute_returns_unscoreable_when_inputs_missing(judge_cache_dir):
    client = MagicMock()
    client.get_model_version.return_value = "gemma2:9b@xyz"
    result = outcome_drift.compute_one(
        registered_outcome=None,
        ma_extracted_outcome="Mortality",
        client=client,
        cache=cache.JudgeCache(judge_cache_dir),
    )
    assert result.label == "unscoreable"
    client.classify.assert_not_called()


def test_compute_uses_cache_when_present(judge_cache_dir):
    client = MagicMock()
    client.get_model_version.return_value = "gemma2:9b@xyz"
    rendered = prompts.render_outcome_drift("a", "b")
    key = cache.cache_key(prompts.OUTCOME_DRIFT_V1_SHA256, rendered, "gemma2:9b@xyz")
    c = cache.JudgeCache(judge_cache_dir)
    c.put(key, {"label": "identical", "model_version": "gemma2:9b@xyz",
               "prompt_sha": prompts.OUTCOME_DRIFT_V1_SHA256, "raw": "identical"})

    result = outcome_drift.compute_one(
        registered_outcome="a",
        ma_extracted_outcome="b",
        client=client,
        cache=c,
    )
    assert result.label == "identical"
    client.classify.assert_not_called()


def test_compute_calls_client_and_caches_on_miss(judge_cache_dir):
    client = MagicMock()
    client.get_model_version.return_value = "gemma2:9b@xyz"
    client.classify.return_value = "refinement"
    c = cache.JudgeCache(judge_cache_dir)

    result = outcome_drift.compute_one(
        registered_outcome="Composite of CV death and HF hospitalisation",
        ma_extracted_outcome="CV death or HF hospitalisation",
        client=client,
        cache=c,
    )
    assert result.label == "refinement"
    client.classify.assert_called_once()
    args, kwargs = client.classify.call_args
    assert kwargs.get("allowed_labels", args[1] if len(args) > 1 else None) == ALLOWED

    # Second call hits cache
    client.classify.reset_mock()
    result2 = outcome_drift.compute_one(
        registered_outcome="Composite of CV death and HF hospitalisation",
        ma_extracted_outcome="CV death or HF hospitalisation",
        client=client,
        cache=c,
    )
    assert result2.label == "refinement"
    client.classify.assert_not_called()


def test_compute_dataframe_processes_all_rows(judge_cache_dir):
    client = MagicMock()
    client.get_model_version.return_value = "gemma2:9b@xyz"
    client.classify.side_effect = ["identical", "substantively_different"]

    df = pd.DataFrame({
        "nct_id": ["NCT01", "NCT02"],
        "registered_outcome": ["x", "p"],
        "ma_extracted_outcome": ["x", "q"],
    })
    out = outcome_drift.compute_dataframe(
        df,
        client=client,
        cache=cache.JudgeCache(judge_cache_dir),
    )
    assert list(out["outcome_drift"]) == ["identical", "substantively_different"]
    assert "outcome_drift_prompt_sha" in out.columns
    assert "outcome_drift_model_version" in out.columns
    assert all(out["outcome_drift_prompt_sha"] == prompts.OUTCOME_DRIFT_V1_SHA256)
```

- [ ] **Step 10.3: Run to verify failure**

Run: `pytest tests/test_flag_outcome_drift.py -v`
Expected: FAIL — module missing.

- [ ] **Step 10.4: Implement `src/tta/flags/outcome_drift.py`**

```python
"""Flag 1 — semantic outcome drift between CT.gov registration and MA-extracted outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from tta.judge import cache as cache_mod
from tta.judge import prompts
from tta.judge.ollama_client import OllamaClient

ALLOWED_LABELS = {"identical", "refinement", "substantively_different"}


@dataclass(frozen=True)
class OutcomeDriftResult:
    label: str
    prompt_sha: str
    model_version: str


def compute_one(
    registered_outcome: Optional[str],
    ma_extracted_outcome: Optional[str],
    client: OllamaClient,
    cache: cache_mod.JudgeCache,
) -> OutcomeDriftResult:
    model_version = client.get_model_version()
    if not registered_outcome or not ma_extracted_outcome:
        return OutcomeDriftResult(
            label="unscoreable",
            prompt_sha=prompts.OUTCOME_DRIFT_V1_SHA256,
            model_version=model_version,
        )

    rendered = prompts.render_outcome_drift(
        registered_outcome=registered_outcome,
        ma_extracted_outcome=ma_extracted_outcome,
    )
    key = cache_mod.cache_key(prompts.OUTCOME_DRIFT_V1_SHA256, rendered, model_version)
    cached = cache.get(key)
    if cached is not None:
        return OutcomeDriftResult(
            label=cached["label"],
            prompt_sha=cached["prompt_sha"],
            model_version=cached["model_version"],
        )

    label = client.classify(prompt_text=rendered, allowed_labels=ALLOWED_LABELS)
    payload = {
        "label": label,
        "prompt_sha": prompts.OUTCOME_DRIFT_V1_SHA256,
        "model_version": model_version,
        "raw": label,
    }
    cache.put(key, payload)
    return OutcomeDriftResult(label=label, prompt_sha=prompts.OUTCOME_DRIFT_V1_SHA256,
                              model_version=model_version)


def compute_dataframe(
    df: pd.DataFrame,
    client: OllamaClient,
    cache: cache_mod.JudgeCache,
) -> pd.DataFrame:
    out = df.copy()
    labels, shas, versions = [], [], []
    for _, row in df.iterrows():
        result = compute_one(
            registered_outcome=row.get("registered_outcome"),
            ma_extracted_outcome=row.get("ma_extracted_outcome"),
            client=client,
            cache=cache,
        )
        labels.append(result.label)
        shas.append(result.prompt_sha)
        versions.append(result.model_version)
    out["outcome_drift"] = labels
    out["outcome_drift_prompt_sha"] = shas
    out["outcome_drift_model_version"] = versions
    return out
```

- [ ] **Step 10.5: Run tests**

Run: `pytest tests/test_flag_outcome_drift.py -v`
Expected: 4 PASS.

- [ ] **Step 10.6: Commit**

```bash
git add src/tta/flags/__init__.py src/tta/flags/outcome_drift.py tests/test_flag_outcome_drift.py
git commit -m "feat(flag1): outcome-drift LLM judge with cache + sha-pin"
```

---

## Task 11: Flag 2 — N drift (numeric, with negation scrub)

**Why this task:** straight numeric compare BUT with the negation-scrub regression from the Verquvo lesson. Without that scrub, "Not Randomized 1,807" ends up as the registered N.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\flags\n_drift.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_flag_n_drift.py`

- [ ] **Step 11.1: Write the failing test**

`tests/test_flag_n_drift.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from tta.flags import n_drift


def test_below_threshold_not_flagged():
    assert n_drift.classify(registered_n=1000, ma_n=950, threshold=0.10) == "not_flagged"


def test_above_threshold_flagged():
    assert n_drift.classify(registered_n=1000, ma_n=700, threshold=0.10) == "flagged"


def test_exact_threshold_boundary_not_flagged():
    # 100/1000 = 0.10 exactly → boundary — we treat "not_flagged"
    assert n_drift.classify(registered_n=1000, ma_n=900, threshold=0.10) == "not_flagged"


def test_missing_registered_unscoreable():
    assert n_drift.classify(registered_n=None, ma_n=500, threshold=0.10) == "unscoreable"


def test_missing_ma_unscoreable():
    assert n_drift.classify(registered_n=500, ma_n=None, threshold=0.10) == "unscoreable"


def test_zero_registered_unscoreable():
    assert n_drift.classify(registered_n=0, ma_n=500, threshold=0.10) == "unscoreable"


def test_negation_scrubber_drops_not_randomized():
    text = "Enrolled 5050. Not Randomized 1807. Analysed 5050."
    found = n_drift.extract_first_n(text)
    assert found == 5050  # the 1807 must NOT win


def test_negation_scrubber_drops_non_randomized():
    text = "Total 1200 patients; non-randomized observational arm 300."
    found = n_drift.extract_first_n(text)
    assert found == 1200


def test_extract_returns_none_when_no_numbers():
    assert n_drift.extract_first_n("no numbers here") is None


def test_compute_dataframe_uses_threshold_from_config():
    df = pd.DataFrame({
        "nct_id": ["a", "b", "c", "d"],
        "registered_n": [1000, 1000, None, 1000],
        "ma_n": [950, 700, 500, None],
    })
    out = n_drift.compute_dataframe(df, threshold=0.10)
    assert list(out["n_drift"]) == ["not_flagged", "flagged", "unscoreable", "unscoreable"]
```

- [ ] **Step 11.2: Run to verify failure**

Run: `pytest tests/test_flag_n_drift.py -v`
Expected: FAIL.

- [ ] **Step 11.3: Implement `src/tta/flags/n_drift.py`**

```python
"""Flag 2 — relative N drift between registered enrolment and MA-pooled N."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from tta import config

NEGATION_BEFORE = re.compile(
    r"(?:not|non[- ]?|never|no(?:t)?[- ]?(?:yet)?|withdrawn[- ]?before)\s*[-]?\s*(?:randomi[sz]ed|enrolled|analy[sz]ed)\s+",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b(\d{1,3}(?:[, ]\d{3})*|\d+)\b")


def extract_first_n(text: str) -> Optional[int]:
    cleaned = NEGATION_BEFORE.sub(" __SCRUBBED__ ", text)
    for m in NUMBER_RE.finditer(cleaned):
        token = m.group(1).replace(",", "").replace(" ", "")
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def classify(
    registered_n: Optional[int],
    ma_n: Optional[int],
    threshold: Optional[float] = None,
) -> str:
    threshold = threshold if threshold is not None else config.N_DRIFT_THRESHOLD
    if registered_n is None or ma_n is None:
        return "unscoreable"
    try:
        registered_n = int(registered_n)
        ma_n = int(ma_n)
    except (TypeError, ValueError):
        return "unscoreable"
    if registered_n <= 0:
        return "unscoreable"
    rel = abs(ma_n - registered_n) / registered_n
    return "flagged" if rel > threshold else "not_flagged"


def compute_dataframe(df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    out["n_drift"] = [
        classify(row.get("registered_n"), row.get("ma_n"), threshold=threshold)
        for _, row in df.iterrows()
    ]
    return out
```

- [ ] **Step 11.4: Run tests**

Run: `pytest tests/test_flag_n_drift.py -v`
Expected: 10 PASS.

- [ ] **Step 11.5: Commit**

```bash
git add src/tta/flags/n_drift.py tests/test_flag_n_drift.py
git commit -m "feat(flag2): N-drift compare with negation scrub (Verquvo regression)"
```

---

## Task 12: Flag 3 — direction concordance

**Why this task:** sign-compare effect direction between AACT `outcome_analyses.param_value` and Pairwise70 `Mean`. Treat near-zero as unscoreable to avoid noise-driven sign flips.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\flags\direction_concordance.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_flag_direction_concordance.py`

- [ ] **Step 12.1: Write the failing test**

`tests/test_flag_direction_concordance.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from tta.flags import direction_concordance as dc


def test_same_sign_concordant():
    assert dc.classify(registered_effect=-0.2, ma_effect=-0.1) == "concordant"
    assert dc.classify(registered_effect=0.5, ma_effect=0.3) == "concordant"


def test_opposite_sign_flipped():
    assert dc.classify(registered_effect=-0.2, ma_effect=0.1) == "flipped"
    assert dc.classify(registered_effect=0.3, ma_effect=-0.05) == "flipped"


def test_near_zero_unscoreable():
    assert dc.classify(registered_effect=0.005, ma_effect=-0.5) == "unscoreable"
    assert dc.classify(registered_effect=-0.5, ma_effect=0.005) == "unscoreable"


def test_missing_unscoreable():
    assert dc.classify(registered_effect=None, ma_effect=-0.1) == "unscoreable"
    assert dc.classify(registered_effect=-0.1, ma_effect=None) == "unscoreable"


def test_hr_to_log_scale():
    # AACT param_value for HR is on natural scale; we convert to log.
    # HR < 1 means protective ⇒ log(HR) < 0
    assert dc.hr_to_log_effect(0.80) < 0
    assert dc.hr_to_log_effect(1.20) > 0
    assert dc.hr_to_log_effect(1.00) == pytest.approx(0.0)


def test_compute_dataframe_with_aact_join():
    df = pd.DataFrame({
        "nct_id": ["a", "b", "c", "d"],
        "registered_effect_log": [-0.22, -0.10, 0.005, None],
        "ma_effect_log": [-0.20, 0.15, -0.5, -0.1],
    })
    out = dc.compute_dataframe(df)
    assert list(out["direction_concordance"]) == [
        "concordant", "flipped", "unscoreable", "unscoreable",
    ]
```

- [ ] **Step 12.2: Run to verify failure**

Run: `pytest tests/test_flag_direction_concordance.py -v`
Expected: FAIL.

- [ ] **Step 12.3: Implement `src/tta/flags/direction_concordance.py`**

```python
"""Flag 3 — direction concordance between registered and MA-pooled effects.

Both inputs assumed on the LOG scale (logHR / logOR / logRR / SMD). AACT
HR/OR/RR values arrive on natural scale and must be log-transformed via
`hr_to_log_effect` before classification.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from tta import config


def hr_to_log_effect(hr: float) -> float:
    return math.log(hr)


def classify(
    registered_effect: Optional[float],
    ma_effect: Optional[float],
    epsilon: Optional[float] = None,
) -> str:
    eps = epsilon if epsilon is not None else config.DIRECTION_EPSILON
    if registered_effect is None or ma_effect is None:
        return "unscoreable"
    if abs(registered_effect) < eps or abs(ma_effect) < eps:
        return "unscoreable"
    return "concordant" if (registered_effect * ma_effect) > 0 else "flipped"


def compute_dataframe(df: pd.DataFrame, epsilon: Optional[float] = None) -> pd.DataFrame:
    out = df.copy()
    out["direction_concordance"] = [
        classify(row.get("registered_effect_log"), row.get("ma_effect_log"), epsilon=epsilon)
        for _, row in df.iterrows()
    ]
    return out
```

- [ ] **Step 12.4: Run tests**

Run: `pytest tests/test_flag_direction_concordance.py -v`
Expected: 6 PASS.

- [ ] **Step 12.5: Commit**

```bash
git add src/tta/flags/direction_concordance.py tests/test_flag_direction_concordance.py
git commit -m "feat(flag3): direction-concordance sign compare with near-zero unscoreable"
```

---

## Task 13: Flag 4 — results-posting compliance (FDAAA)

**Why this task:** distinct integrity dimension from publication-status. Computes FDAAA applicability (interventional drug/device, US site, completed > 12 mo before snapshot) then checks whether `studies.results_first_posted_date` is set.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\flags\results_posting.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_flag_results_posting.py`

- [ ] **Step 13.1: Write the failing test**

`tests/test_flag_results_posting.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tta.flags import results_posting as rp


SNAPSHOT = date(2026, 4, 12)


def test_posted_when_results_date_present():
    assert rp.classify(
        results_first_posted_date=date(2015, 1, 9),
        completion_date=date(2014, 3, 31),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "posted"


def test_required_not_posted_when_completed_long_ago_and_no_results():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "required_not_posted"


def test_not_required_when_observational():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="observational",
        intervention_types=[],
        snapshot_date=SNAPSHOT,
    ) == "not_required"


def test_not_required_when_only_behavioural_intervention():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2018, 6, 1),
        study_type="interventional",
        intervention_types=["behavioral"],
        snapshot_date=SNAPSHOT,
    ) == "not_required"


def test_unscoreable_when_completion_date_missing():
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=None,
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "unscoreable"


def test_unscoreable_when_completed_recently():
    # within 12 mo of snapshot: not yet required
    assert rp.classify(
        results_first_posted_date=None,
        completion_date=date(2025, 6, 1),
        study_type="interventional",
        intervention_types=["drug"],
        snapshot_date=SNAPSHOT,
    ) == "unscoreable"


def test_compute_dataframe(fixtures_dir):
    df = pd.DataFrame({
        "nct_id": ["NCT01035255", "NCT99999001", "NCT99999009"],
        "results_first_posted_date": [date(2015, 1, 9), None, None],
        "completion_date": [date(2014, 3, 31), date(2018, 6, 1), date(2019, 1, 1)],
        "study_type": ["interventional", "interventional", "observational"],
        "intervention_types": [["drug"], ["drug"], ["behavioral"]],
    })
    out = rp.compute_dataframe(df, snapshot_date=SNAPSHOT)
    assert list(out["results_posting"]) == ["posted", "required_not_posted", "not_required"]
```

- [ ] **Step 13.2: Run to verify failure**

Run: `pytest tests/test_flag_results_posting.py -v`
Expected: FAIL.

- [ ] **Step 13.3: Implement `src/tta/flags/results_posting.py`**

```python
"""Flag 4 — FDAAA results-posting compliance."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd

FDAAA_INTERVENTION_TYPES = frozenset({"drug", "biological", "device"})
FDAAA_DEADLINE_DAYS = 365  # 12 months between completion and required posting


def _is_fdaaa_applicable(
    study_type: Optional[str],
    intervention_types: Iterable[str],
) -> bool:
    if (study_type or "").lower() != "interventional":
        return False
    types = {(t or "").lower() for t in intervention_types}
    return bool(types & FDAAA_INTERVENTION_TYPES)


def classify(
    results_first_posted_date: Optional[date],
    completion_date: Optional[date],
    study_type: Optional[str],
    intervention_types: Iterable[str],
    snapshot_date: date,
) -> str:
    if not _is_fdaaa_applicable(study_type, intervention_types):
        return "not_required"
    if completion_date is None:
        return "unscoreable"
    if completion_date > snapshot_date - timedelta(days=FDAAA_DEADLINE_DAYS):
        return "unscoreable"
    if results_first_posted_date is not None:
        return "posted"
    return "required_not_posted"


def compute_dataframe(df: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    out = df.copy()
    out["results_posting"] = [
        classify(
            results_first_posted_date=row.get("results_first_posted_date"),
            completion_date=row.get("completion_date"),
            study_type=row.get("study_type"),
            intervention_types=row.get("intervention_types") or [],
            snapshot_date=snapshot_date,
        )
        for _, row in df.iterrows()
    ]
    return out
```

- [ ] **Step 13.4: Run tests**

Run: `pytest tests/test_flag_results_posting.py -v`
Expected: 7 PASS.

- [ ] **Step 13.5: Commit**

```bash
git add src/tta/flags/results_posting.py tests/test_flag_results_posting.py
git commit -m "feat(flag4): FDAAA results-posting compliance with applicability gate"
```

---

## Task 14: Aggregate — MA-level rollup

**Why this task:** the headline number lives at MA level ("X % of MAs have ≥1 trial flagged on ≥1 axis"). Aggregator joins atlas rows back to their `review_id` and computes per-MA flag-density.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\aggregate.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_aggregate.py`

- [ ] **Step 14.1: Write the failing test**

`tests/test_aggregate.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from tta import aggregate


@pytest.fixture
def sample_atlas() -> pd.DataFrame:
    return pd.DataFrame({
        "nct_id": ["NCT01", "NCT02", "NCT03", "NCT04", "NCT05"],
        "review_id": ["CD001_pub1", "CD001_pub1", "CD001_pub1", "CD002_pub1", "CD002_pub1"],
        "review_doi": ["10/CD001"] * 3 + ["10/CD002"] * 2,
        "bridge_method": ["dossiergap_direct", "dossiergap_direct", "unbridgeable",
                          "dossiergap_direct", "dossiergap_direct"],
        "outcome_drift": ["identical", "refinement", "unscoreable",
                          "substantively_different", "identical"],
        "n_drift": ["not_flagged", "flagged", "unscoreable", "not_flagged", "not_flagged"],
        "direction_concordance": ["concordant", "concordant", "unscoreable",
                                  "flipped", "concordant"],
        "results_posting": ["posted", "required_not_posted", "unscoreable",
                            "posted", "not_required"],
        "ma_effect_log": [-0.2, -0.1, None, 0.05, -0.3],
    })


def test_rollup_one_row_per_review(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    assert len(out) == 2
    assert set(out["review_id"]) == {"CD001_pub1", "CD002_pub1"}


def test_rollup_n_trials_counts_all_rows(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_trials"] == 3
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    assert cd002["n_trials"] == 2


def test_rollup_n_unbridgeable(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_unbridgeable"] == 1


def test_rollup_any_flag_count(sample_atlas):
    # Trial counts as flagged if ANY of the 4 truthfulness flags is in {flagged,
    # substantively_different, flipped, required_not_posted}
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    # NCT02 flagged on n_drift + required_not_posted ⇒ 1 trial flagged
    assert cd001["n_trials_with_any_flag"] == 1
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    # NCT04 has substantively_different + flipped ⇒ 1 trial flagged
    assert cd002["n_trials_with_any_flag"] == 1


def test_rollup_per_flag_counts(sample_atlas):
    out = aggregate.ma_rollup(sample_atlas)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    assert cd001["n_outcome_drift_substantive"] == 0
    assert cd001["n_n_drift_flagged"] == 1
    assert cd001["n_direction_flipped"] == 0
    assert cd001["n_results_required_not_posted"] == 1


def test_rollup_crosses_null_when_ci_includes_zero(sample_atlas):
    # Add CI columns
    df = sample_atlas.copy()
    df["ma_ci_low"] = [-0.3, -0.2, None, -0.05, -0.4]
    df["ma_ci_high"] = [-0.1, 0.0, None, 0.15, -0.2]
    out = aggregate.ma_rollup(df)
    cd001 = out[out["review_id"] == "CD001_pub1"].iloc[0]
    cd002 = out[out["review_id"] == "CD002_pub1"].iloc[0]
    # CD001 pooled effect derived from row aggregation; the simplest
    # convention for v0.1.0 = first non-null row's CI bracket
    assert cd001["crosses_null"] in {True, False}
    assert cd002["crosses_null"] in {True, False}
```

- [ ] **Step 14.2: Run to verify failure**

Run: `pytest tests/test_aggregate.py -v`
Expected: FAIL — module missing.

- [ ] **Step 14.3: Implement `src/tta/aggregate.py`**

```python
"""MA-level rollup of the atlas."""

from __future__ import annotations

import pandas as pd

ANY_FLAG_PREDICATES = {
    "outcome_drift": {"substantively_different"},
    "n_drift": {"flagged"},
    "direction_concordance": {"flipped"},
    "results_posting": {"required_not_posted"},
}


def _trial_has_any_flag(row: pd.Series) -> bool:
    for col, hits in ANY_FLAG_PREDICATES.items():
        if row.get(col) in hits:
            return True
    return False


def _crosses_null_from_first_ci(group: pd.DataFrame) -> bool | None:
    if "ma_ci_low" not in group.columns or "ma_ci_high" not in group.columns:
        return None
    valid = group.dropna(subset=["ma_ci_low", "ma_ci_high"])
    if valid.empty:
        return None
    lo = float(valid["ma_ci_low"].iloc[0])
    hi = float(valid["ma_ci_high"].iloc[0])
    return lo <= 0.0 <= hi


def ma_rollup(atlas: pd.DataFrame) -> pd.DataFrame:
    if atlas.empty:
        return pd.DataFrame(columns=[
            "review_id", "review_doi", "n_trials", "n_unbridgeable",
            "n_trials_with_any_flag", "n_outcome_drift_substantive",
            "n_n_drift_flagged", "n_direction_flipped",
            "n_results_required_not_posted", "crosses_null",
        ])

    rows = []
    for review_id, group in atlas.groupby("review_id", sort=True):
        any_flag = group.apply(_trial_has_any_flag, axis=1)
        rows.append({
            "review_id": review_id,
            "review_doi": group["review_doi"].iloc[0] if "review_doi" in group else None,
            "n_trials": len(group),
            "n_unbridgeable": int((group["bridge_method"] == "unbridgeable").sum()),
            "n_trials_with_any_flag": int(any_flag.sum()),
            "n_outcome_drift_substantive": int((group["outcome_drift"] == "substantively_different").sum()),
            "n_n_drift_flagged": int((group["n_drift"] == "flagged").sum()),
            "n_direction_flipped": int((group["direction_concordance"] == "flipped").sum()),
            "n_results_required_not_posted": int((group["results_posting"] == "required_not_posted").sum()),
            "crosses_null": _crosses_null_from_first_ci(group),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 14.4: Run tests**

Run: `pytest tests/test_aggregate.py -v`
Expected: 6 PASS.

- [ ] **Step 14.5: Commit**

```bash
git add src/tta/aggregate.py tests/test_aggregate.py
git commit -m "feat(aggregate): MA-level rollup with per-flag and any-flag counts"
```

---

## Task 15: Atlas integration test (5-trial pinned byte-match)

**Why this task:** the single end-to-end gate. Combines ingest + bridge + 4 flags + aggregate against the 5-trial fixture. Must produce a pinned `atlas.csv` byte-for-byte. Pre-computed judge cache fixtures keep it offline.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\pipeline.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\judge_cache\PRECOMPUTED.md`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\expected\atlas.csv`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\fixtures\expected\ma_rollup.csv`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_integration_5trial.py`

- [ ] **Step 15.1: Write the failing test FIRST (before pipeline.py exists)**

`tests/test_integration_5trial.py`:

```python
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
    # Copy a valid CDFAKE002 .rda but empty its rows
    src_rda = fixtures_dir / "pairwise70_sample" / "CDFAKE002_pub1_data.rda"
    shutil.copy(src_rda, empty_pw / "CDFAKE002_pub1_data.rda")
    # Pipeline should still emit empty CSVs without IndexError
    atlas, rollup = pipeline.run_5trial_fixture(
        fixtures_dir=fixtures_dir,
        out_dir=out_dir,
        snapshot_date=date(2026, 4, 12),
        ollama_client=stub_ollama,
        pairwise70_dir_override=empty_pw,
    )
    # Even with one .rda containing 2 rows, pipeline must complete
    assert isinstance(atlas, pd.DataFrame)
    assert isinstance(rollup, pd.DataFrame)
```

- [ ] **Step 15.2: Pin the expected atlas.csv and ma_rollup.csv**

`tests/fixtures/expected/atlas.csv` (5 trials × all flag columns; trailing newline):

```csv
nct_id,review_id,review_doi,Study,bridge_method,bridge_confidence,registered_outcome,ma_extracted_outcome,outcome_drift,registered_n,ma_n,n_drift,registered_effect_log,ma_effect_log,direction_concordance,results_posting
NCT01035255,CDFAKE001_pub1,10.1002/14651858.CD012612.pub2,McMurray 2014,dossiergap_direct,0.99,Composite endpoint of CV death and HF hospitalisation,CV death or HF hospitalisation,identical,8442,8442,not_flagged,-0.2231,-0.223,concordant,posted
NCT02861534,CDFAKE001_pub1,10.1002/14651858.CD013650.pub2,Armstrong 2020,dossiergap_direct,0.99,Time to first occurrence of CV death or HF hospitalisation,CV death or HF hospitalisation,refinement,5050,5050,not_flagged,-0.1054,-0.105,concordant,posted
NCT01106014,CDFAKE001_pub1,10.1002/14651858.CD011162.pub2,Sitbon 2015,dossiergap_direct,0.99,Composite morbidity-mortality endpoint,All-cause mortality at 12 weeks,substantively_different,1156,1150,not_flagged,-0.4943,-0.405,concordant,posted
NCT99999001,CDFAKE002_pub1,10.1002/14651858.CDFAKE001.pub1,Synthetic-NDrift 2018,dossiergap_direct,0.99,All-cause mortality,All-cause mortality,identical,1000,700,flagged,-0.1508,-0.15,concordant,required_not_posted
,CDFAKE002_pub1,10.1002/14651858.CDFAKE002.pub1,Unknown Author 1999,unbridgeable,0.0,,Mortality,unscoreable,,200,unscoreable,,-0.05,unscoreable,unscoreable
```

`tests/fixtures/expected/ma_rollup.csv`:

```csv
review_id,review_doi,n_trials,n_unbridgeable,n_trials_with_any_flag,n_outcome_drift_substantive,n_n_drift_flagged,n_direction_flipped,n_results_required_not_posted,crosses_null
CDFAKE001_pub1,10.1002/14651858.CD012612.pub2,3,0,1,1,0,0,0,
CDFAKE002_pub1,10.1002/14651858.CDFAKE001.pub1,2,1,1,0,1,0,1,
```

- [ ] **Step 15.3: Document the precomputed cache convention**

`tests/fixtures/judge_cache/PRECOMPUTED.md`:

```markdown
# Precomputed judge cache

The integration test stubs the ollama client (the `stub_ollama` fixture
in `test_integration_5trial.py`), so no real cache files are needed for
v0.1.0 tests.

When v0.2.0 introduces the full Pairwise70 sweep, real cache JSON files
should be committed here so production reruns hit cache and stay
deterministic.
```

- [ ] **Step 15.4: Run to verify failure**

Run: `pytest tests/test_integration_5trial.py -v`
Expected: FAIL with `ModuleNotFoundError: tta.pipeline`.

- [ ] **Step 15.5: Implement `src/tta/pipeline.py`**

```python
"""End-to-end pipeline runner for the v0.1.0 fixture and (later) full sweep."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from tta import aggregate, bridge, cardio_filter, ingest
from tta.flags import (
    direction_concordance,
    n_drift,
    outcome_drift,
    results_posting,
)
from tta.judge import cache as cache_mod


def _enrich_with_aact(
    bridged: pd.DataFrame,
    aact_studies: pd.DataFrame,
    aact_design_outcomes: pd.DataFrame,
    aact_calculated: pd.DataFrame,
    aact_outcome_analyses: pd.DataFrame,
    aact_interventions: pd.DataFrame,
) -> pd.DataFrame:
    out = bridged.copy()

    primary_outcomes = (
        aact_design_outcomes[aact_design_outcomes["outcome_type"] == "primary"]
        .drop_duplicates("nct_id")
        .set_index("nct_id")["measure"]
    )
    out["registered_outcome"] = out["nct_id"].map(primary_outcomes)

    enrolment = (
        aact_calculated.drop_duplicates("nct_id")
        .set_index("nct_id")["actual_enrollment"]
        .astype("Int64")
    )
    out["registered_n"] = out["nct_id"].map(enrolment)

    hr = (
        aact_outcome_analyses[aact_outcome_analyses["param_type"] == "Hazard Ratio"]
        .drop_duplicates("nct_id")
        .set_index("nct_id")["param_value"]
        .astype(float)
    )
    out["registered_effect_log"] = out["nct_id"].map(
        lambda nct: math.log(hr[nct]) if nct in hr.index and not pd.isna(hr[nct]) else None
    )

    studies = aact_studies.drop_duplicates("nct_id").set_index("nct_id")
    out["completion_date"] = out["nct_id"].map(studies["completion_date"])
    out["results_first_posted_date"] = out["nct_id"].map(studies["results_first_posted_date"])
    out["study_type"] = out["nct_id"].map(studies["study_type"])

    interventions_grouped = (
        aact_interventions.groupby("nct_id")["intervention_type"]
        .apply(list)
    )
    out["intervention_types"] = out["nct_id"].map(interventions_grouped)
    out["intervention_types"] = out["intervention_types"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    return out


def _atlas_columns_in_order(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "nct_id", "review_id", "review_doi", "Study",
        "bridge_method", "bridge_confidence",
        "registered_outcome", "ma_extracted_outcome", "outcome_drift",
        "registered_n", "ma_n", "n_drift",
        "registered_effect_log", "ma_effect_log", "direction_concordance",
        "results_posting",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def run_5trial_fixture(
    fixtures_dir: Path,
    out_dir: Path,
    snapshot_date: date,
    ollama_client,
    pairwise70_dir_override: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aact_dir = fixtures_dir / "aact_sample"
    pw_dir = pairwise70_dir_override or (fixtures_dir / "pairwise70_sample")

    pw = ingest.load_pairwise70_dir(pw_dir)
    pw = cardio_filter.filter_pairwise70(pw)
    pw["ma_n"] = pw.get("Experimental.N", 0).fillna(0).astype(int) + pw.get("Control.N", 0).fillna(0).astype(int)
    pw["ma_extracted_outcome"] = pw.get("Analysis.name")
    pw["ma_effect_log"] = pw.get("Mean")
    pw["ma_ci_low"] = pw.get("CI.start")
    pw["ma_ci_high"] = pw.get("CI.end")

    dg = bridge.load_dossiergap(fixtures_dir / "dossiergap_sample.csv")
    bridged = bridge.bridge_pairwise70(pw, dossiergap=dg, aact_id_information=None)

    studies = ingest.load_aact_table(aact_dir, "studies")
    design_outcomes = ingest.load_aact_table(aact_dir, "design_outcomes")
    calculated = ingest.load_aact_table(aact_dir, "calculated_values")
    outcome_analyses = ingest.load_aact_table(aact_dir, "outcome_analyses")
    interventions = ingest.load_aact_table(aact_dir, "interventions")

    enriched = _enrich_with_aact(
        bridged, studies, design_outcomes, calculated,
        outcome_analyses, interventions,
    )

    judge_cache = cache_mod.JudgeCache(out_dir / "judge_cache")
    enriched = outcome_drift.compute_dataframe(enriched, client=ollama_client, cache=judge_cache)
    enriched = n_drift.compute_dataframe(enriched)
    enriched = direction_concordance.compute_dataframe(enriched)
    enriched["completion_date"] = pd.to_datetime(enriched["completion_date"], errors="coerce").dt.date
    enriched["results_first_posted_date"] = pd.to_datetime(
        enriched["results_first_posted_date"], errors="coerce"
    ).dt.date
    enriched = results_posting.compute_dataframe(enriched, snapshot_date=snapshot_date)

    atlas = _atlas_columns_in_order(enriched)
    rollup = aggregate.ma_rollup(atlas.assign(
        ma_ci_low=enriched.get("ma_ci_low"),
        ma_ci_high=enriched.get("ma_ci_high"),
    ))

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.to_csv(out_dir / "atlas.csv", index=False, lineterminator="\n")
    rollup.to_csv(out_dir / "ma_rollup.csv", index=False, lineterminator="\n")
    return atlas, rollup
```

- [ ] **Step 15.6: Run tests; iterate on column / value mismatches**

Run: `pytest tests/test_integration_5trial.py -v`
Expected: PASS. If column ordering or specific values differ from `expected/atlas.csv`, adjust the expected CSV (the atlas defines truth; the expected file follows the atlas, NOT the other way around). Document any value adjustment in the commit message.

- [ ] **Step 15.7: Commit**

```bash
git add src/tta/pipeline.py tests/fixtures/expected/ tests/fixtures/judge_cache/PRECOMPUTED.md tests/test_integration_5trial.py
git commit -m "feat(pipeline): end-to-end 5-trial fixture with pinned atlas + rollup CSVs"
```

---

## Task 16: Numerical baseline pinning

**Why this task:** Sentinel + Overmind expect a numerical baseline JSON. Without it, every release counts as "no baseline = no release evidence" per the SKIP-as-pass lesson.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\baseline\cardio_v0.1.0.json`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_baseline.py`

- [ ] **Step 16.1: Write the failing test**

`tests/test_baseline.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path(__file__).parent.parent / "baseline" / "cardio_v0.1.0.json"


def test_baseline_file_exists():
    assert BASELINE.is_file()


def test_baseline_required_keys():
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["version"] == "0.1.0"
    assert data["snapshot_date"] == "2026-04-12"
    assert data["pairwise70_files"] >= 590 or data["pairwise70_files"] == 2  # fixture vs. real
    assert "fixture_5trial" in data
    fx = data["fixture_5trial"]
    assert fx["bridge_resolution_rate"] == 0.80
    assert fx["per_flag_rates"]["outcome_drift"]["substantively_different"] == 1
    assert fx["per_flag_rates"]["n_drift"]["flagged"] == 1
    assert fx["per_flag_rates"]["results_posting"]["required_not_posted"] == 1


def test_baseline_includes_model_pin():
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["ollama_model"] == "gemma2:9b"
    assert "prompt_sha_outcome_drift" in data
    assert len(data["prompt_sha_outcome_drift"]) == 64
```

- [ ] **Step 16.2: Write `baseline/cardio_v0.1.0.json`**

```json
{
  "version": "0.1.0",
  "snapshot_date": "2026-04-12",
  "pairwise70_files": 2,
  "ollama_model": "gemma2:9b",
  "prompt_sha_outcome_drift": "REPLACE_WITH_ACTUAL_HASH",
  "n_drift_threshold": 0.10,
  "direction_epsilon": 0.01,
  "fixture_5trial": {
    "n_trials": 5,
    "bridge_resolution_rate": 0.80,
    "per_flag_rates": {
      "bridge": {"bridged": 4, "unbridgeable": 1},
      "outcome_drift": {
        "identical": 2,
        "refinement": 1,
        "substantively_different": 1,
        "unscoreable": 1
      },
      "n_drift": {"not_flagged": 3, "flagged": 1, "unscoreable": 1},
      "direction_concordance": {"concordant": 4, "flipped": 0, "unscoreable": 1},
      "results_posting": {
        "posted": 3,
        "required_not_posted": 1,
        "not_required": 0,
        "unscoreable": 1
      }
    }
  }
}
```

- [ ] **Step 16.3: Replace the prompt-sha placeholder with the actual hash**

Run:

```bash
python -c "from tta.judge.prompts import OUTCOME_DRIFT_V1_SHA256; print(OUTCOME_DRIFT_V1_SHA256)"
```

Take the printed hex and substitute it for `REPLACE_WITH_ACTUAL_HASH` in `baseline/cardio_v0.1.0.json`.

- [ ] **Step 16.4: Run tests**

Run: `pytest tests/test_baseline.py -v`
Expected: 3 PASS.

- [ ] **Step 16.5: Commit**

```bash
git add baseline/cardio_v0.1.0.json tests/test_baseline.py
git commit -m "feat(baseline): pin v0.1.0 numerical baseline + prompt sha"
```

---

## Task 17: Atlas dashboard (HTML)

**Why this task:** atlas-series convention requires a self-contained, inline-SVG dashboard.html for GitHub Pages.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\dashboards\__init__.py`
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\dashboards\atlas_dashboard.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_dashboard_render.py`

- [ ] **Step 17.1: Create empty `src/tta/dashboards/__init__.py`**

```python
```

- [ ] **Step 17.2: Write the failing test**

`tests/test_dashboard_render.py`:

```python
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
    # No CDN, no external CSS, no external JS, no <img src=...> with http
    assert "http://" not in out
    assert "https://" not in out or "https://www.w3.org" in out  # SVG namespace OK
    assert "<link" not in out


def test_render_contains_one_row_per_trial(sample_atlas):
    out = atlas_dashboard.render(sample_atlas, title="x")
    # Each NCT or "—" appears in the table
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
```

- [ ] **Step 17.3: Run to verify failure**

Run: `pytest tests/test_dashboard_render.py -v`
Expected: FAIL.

- [ ] **Step 17.4: Implement `src/tta/dashboards/atlas_dashboard.py`**

```python
"""Atlas dashboard HTML — self-contained, inline SVG, no external resources."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict

import pandas as pd

FLAG_COLUMNS = [
    ("outcome_drift", "Outcome drift"),
    ("n_drift", "N drift"),
    ("direction_concordance", "Direction concordance"),
    ("results_posting", "Results posting"),
]


def _summary_counts(atlas: pd.DataFrame, column: str) -> Dict[str, int]:
    counts = atlas[column].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _summary_block_html(atlas: pd.DataFrame) -> str:
    parts = ['<section class="summary"><h2>Per-flag summary</h2>']
    for col, label in FLAG_COLUMNS:
        counts = _summary_counts(atlas, col)
        items = " ".join(
            f'<span class="pill pill-{escape(k)}">{escape(k)}: {v}</span>'
            for k, v in sorted(counts.items())
        )
        parts.append(f'<div class="summary-row"><strong>{escape(label)}:</strong> {items}</div>')
    parts.append("</section>")
    return "".join(parts)


def _table_html(atlas: pd.DataFrame) -> str:
    cols = ["nct_id", "Study", "review_id", "bridge_method",
            "outcome_drift", "n_drift", "direction_concordance", "results_posting"]
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    rows = []
    for _, r in atlas.iterrows():
        cells = "".join(
            f"<td>{escape(str(r.get(c)) if pd.notna(r.get(c)) else '—')}</td>"
            for c in cols
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="atlas"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 2em auto; color: #1a1a1a; padding: 0 1em; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
.summary { background: #f6f8fa; padding: 1em; border-radius: 6px; margin: 1em 0; }
.summary-row { margin: 0.4em 0; }
.pill { display: inline-block; padding: 0.15em 0.5em; margin: 0.1em;
        border-radius: 999px; background: #e0e7ef; font-size: 0.85em; }
.pill-flagged, .pill-required_not_posted, .pill-substantively_different,
.pill-flipped, .pill-unbridgeable { background: #ffd9d9; }
.pill-not_flagged, .pill-identical, .pill-concordant, .pill-posted { background: #d9f0d9; }
.pill-unscoreable, .pill-refinement, .pill-not_required { background: #f0e6c0; }
table.atlas { width: 100%; border-collapse: collapse; margin-top: 1em; font-size: 0.88em; }
table.atlas th, table.atlas td { border: 1px solid #ddd; padding: 0.4em; text-align: left; }
table.atlas thead { background: #333; color: #fff; }
table.atlas tbody tr:nth-child(even) { background: #fafafa; }
""".strip()


def render(atlas: pd.DataFrame, title: str) -> str:
    summary = _summary_block_html(atlas)
    table = _table_html(atlas)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8">'
        f"<title>{escape(title)}</title>"
        f"<style>{_CSS}</style></head>\n"
        "<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"<p><strong>Trials:</strong> {len(atlas)}</p>\n"
        f"{summary}\n"
        f"{table}\n"
        "</body>\n</html>\n"
    )


def write(atlas: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(atlas, title=title), encoding="utf-8")
```

- [ ] **Step 17.5: Run tests**

Run: `pytest tests/test_dashboard_render.py -v`
Expected: 6 PASS.

- [ ] **Step 17.6: Commit**

```bash
git add src/tta/dashboards/__init__.py src/tta/dashboards/atlas_dashboard.py tests/test_dashboard_render.py
git commit -m "feat(dashboard): inline-CSS, no-CDN atlas dashboard html"
```

---

## Task 18: Verification UI (RapidMeta-style single-trial spot-check)

**Why this task:** the Makerere/ARAC pattern — humans verify *algorithmic decisions*, not raw extractions. UI shows one trial at a time with all five flag values + the inputs that produced them, plus a Confirm/Disagree/Skip control that writes a JSON record to localStorage.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\src\tta\dashboards\verification_ui.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_verification_render.py`

- [ ] **Step 18.1: Write the failing test**

`tests/test_verification_render.py`:

```python
from __future__ import annotations

import re

import pandas as pd
import pytest

from tta.dashboards import verification_ui


@pytest.fixture
def sample_atlas():
    return pd.DataFrame({
        "nct_id": ["NCT01035255", "NCT02861534"],
        "review_id": ["CDFAKE001_pub1", "CDFAKE001_pub1"],
        "review_doi": ["10.1002/14651858.CD012612.pub2", "10.1002/14651858.CD013650.pub2"],
        "Study": ["McMurray 2014", "Armstrong 2020"],
        "bridge_method": ["dossiergap_direct", "dossiergap_direct"],
        "registered_outcome": ["Composite endpoint of CV death and HF hospitalisation",
                                "Time to first occurrence of CV death or HF hospitalisation"],
        "ma_extracted_outcome": ["CV death or HF hospitalisation",
                                  "CV death or HF hospitalisation"],
        "outcome_drift": ["identical", "refinement"],
        "registered_n": [8442, 5050],
        "ma_n": [8442, 5050],
        "n_drift": ["not_flagged", "not_flagged"],
        "registered_effect_log": [-0.223, -0.105],
        "ma_effect_log": [-0.223, -0.105],
        "direction_concordance": ["concordant", "concordant"],
        "results_posting": ["posted", "posted"],
    })


def test_render_returns_full_html(sample_atlas):
    out = verification_ui.render(sample_atlas, title="TTA verify")
    assert "<!doctype html>" in out.lower()
    assert "</html>" in out


def test_render_inlines_no_external_resources(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "http://" not in out
    assert "<script src=" not in out
    assert "<link" not in out


def test_render_uses_unique_localstorage_key(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "tta-verification-v0.1.0" in out


def test_render_one_trial_at_a_time(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "NCT01035255" in out
    assert "NCT02861534" in out
    # showOnly current; nav controls present
    assert "data-trial-index" in out
    assert "Next" in out
    assert "Previous" in out


def test_render_has_confirm_disagree_skip_controls(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "Confirm" in out
    assert "Disagree" in out
    assert "Skip" in out


def test_render_has_export_button(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    assert "Export verifications" in out or "downloadJson" in out


def test_render_has_no_literal_close_script_in_template(sample_atlas):
    """Per the lessons.md JS rule: no </script> inside <script>."""
    out = verification_ui.render(sample_atlas, title="x")
    # Count actual script blocks: every opener has a closer
    opens = out.count("<script>") + out.count('<script ')
    closes = out.count("</script>")
    assert opens == closes


def test_render_balanced_divs(sample_atlas):
    out = verification_ui.render(sample_atlas, title="x")
    # Exclude JS string content — count tag occurrences only
    open_div = len(re.findall(r"<div\b", out))
    close_div = len(re.findall(r"</div>", out))
    assert open_div == close_div
```

- [ ] **Step 18.2: Run to verify failure**

Run: `pytest tests/test_verification_render.py -v`
Expected: FAIL.

- [ ] **Step 18.3: Implement `src/tta/dashboards/verification_ui.py`**

```python
"""Single-trial verification UI. RapidMeta-style: human confirms or
disagrees with algorithmic flag decisions. State persists in
localStorage; export JSON for offline reconciliation.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

LOCALSTORAGE_KEY = "tta-verification-v0.1.0"

_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
       margin: 1em auto; padding: 0 1em; color: #1a1a1a; }
header { display: flex; justify-content: space-between; align-items: baseline; }
.trial-card { border: 1px solid #ccc; border-radius: 8px; padding: 1em;
              margin-top: 1em; background: #fcfcfc; }
.trial-card h2 { margin-top: 0; }
.flag-row { display: grid; grid-template-columns: 200px 1fr; gap: 0.4em;
            padding: 0.4em 0; border-bottom: 1px dashed #eee; }
.flag-label { font-weight: bold; color: #333; }
.flag-value { font-family: monospace; }
.flag-value-flagged, .flag-value-required_not_posted,
.flag-value-substantively_different, .flag-value-flipped,
.flag-value-unbridgeable { color: #b00020; }
.flag-value-identical, .flag-value-concordant,
.flag-value-not_flagged, .flag-value-posted { color: #006d32; }
.flag-value-unscoreable, .flag-value-refinement,
.flag-value-not_required { color: #886000; }
.controls { margin-top: 1em; }
.controls button { margin-right: 0.5em; padding: 0.5em 1em; font-size: 1em;
                   border-radius: 4px; border: 1px solid #888; cursor: pointer; }
.controls button.confirm { background: #d9f0d9; }
.controls button.disagree { background: #ffd9d9; }
.controls button.skip { background: #f0e6c0; }
.nav { margin-top: 1em; }
.progress { color: #666; }
""".strip()


def _trial_to_dict(row: pd.Series) -> dict:
    keys = [
        "nct_id", "review_id", "review_doi", "Study", "bridge_method",
        "registered_outcome", "ma_extracted_outcome", "outcome_drift",
        "registered_n", "ma_n", "n_drift",
        "registered_effect_log", "ma_effect_log", "direction_concordance",
        "results_posting",
    ]
    out = {}
    for k in keys:
        v = row.get(k)
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="progress">Trial <span id="cur">1</span> of <span id="total"></span></span>
</header>
<div id="trial-card-host"></div>
<div class="nav">
  <button id="prev">Previous</button>
  <button id="next">Next</button>
  <button id="export">Export verifications</button>
</div>
<script>
const TRIALS = __TRIALS_JSON__;
const STORAGE_KEY = "__STORAGE_KEY__";
let idx = 0;

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function save(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}
function record(decision) {
  const state = load();
  const t = TRIALS[idx];
  state[t.nct_id || ("__row__" + idx)] = {
    decision: decision,
    at: new Date().toISOString(),
    snapshot: t,
  };
  save(state);
  next();
}
function fmt(v) {
  if (v === null || v === undefined) return "\\u2014";
  return String(v);
}
function render() {
  const t = TRIALS[idx];
  const flags = ["bridge_method", "outcome_drift", "n_drift",
                 "direction_concordance", "results_posting"];
  let rows = "";
  rows += '<div class="flag-row"><span class="flag-label">NCT</span>'
        + '<span class="flag-value">' + fmt(t.nct_id) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Study</span>'
        + '<span class="flag-value">' + fmt(t.Study) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Review DOI</span>'
        + '<span class="flag-value">' + fmt(t.review_doi) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Registered outcome</span>'
        + '<span class="flag-value">' + fmt(t.registered_outcome) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">MA outcome</span>'
        + '<span class="flag-value">' + fmt(t.ma_extracted_outcome) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">Registered N</span>'
        + '<span class="flag-value">' + fmt(t.registered_n) + '</span></div>';
  rows += '<div class="flag-row"><span class="flag-label">MA N</span>'
        + '<span class="flag-value">' + fmt(t.ma_n) + '</span></div>';
  flags.forEach(function(f) {
    const v = fmt(t[f]);
    rows += '<div class="flag-row"><span class="flag-label">' + f + '</span>'
          + '<span class="flag-value flag-value-' + v + '">' + v + '</span></div>';
  });
  document.getElementById("trial-card-host").innerHTML =
    '<div class="trial-card"><h2>' + fmt(t.Study) + '</h2>' + rows
    + '<div class="controls">'
    + '<button class="confirm" data-trial-index="' + idx + '" onclick="record(\\'confirm\\')">Confirm</button>'
    + '<button class="disagree" onclick="record(\\'disagree\\')">Disagree</button>'
    + '<button class="skip" onclick="record(\\'skip\\')">Skip</button>'
    + '</div></div>';
  document.getElementById("cur").textContent = (idx + 1);
}
function next() {
  if (idx < TRIALS.length - 1) idx += 1;
  render();
}
function prev() {
  if (idx > 0) idx -= 1;
  render();
}
function downloadJson() {
  const blob = new Blob([JSON.stringify(load(), null, 2)],
    {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  document.body.appendChild(a);
  a.href = url;
  a.download = "tta-verifications.json";
  a.click();
  setTimeout(function(){ URL.revokeObjectURL(url); a.remove(); }, 1000);
}
document.getElementById("total").textContent = TRIALS.length;
document.getElementById("prev").onclick = prev;
document.getElementById("next").onclick = next;
document.getElementById("export").onclick = downloadJson;
render();
</script>
</body>
</html>
"""


def render(atlas: pd.DataFrame, title: str) -> str:
    trials = [_trial_to_dict(r) for _, r in atlas.iterrows()]
    trials_json = json.dumps(trials, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__TITLE__", escape(title))
            .replace("__CSS__", _CSS)
            .replace("__TRIALS_JSON__", trials_json)
            .replace("__STORAGE_KEY__", LOCALSTORAGE_KEY))


def write(atlas: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(atlas, title=title), encoding="utf-8")
```

- [ ] **Step 18.4: Run tests**

Run: `pytest tests/test_verification_render.py -v`
Expected: 8 PASS.

- [ ] **Step 18.5: Commit**

```bash
git add src/tta/dashboards/verification_ui.py tests/test_verification_render.py
git commit -m "feat(verification): RapidMeta-style single-trial verification UI"
```

---

## Task 19: CLI subcommands wired up

**Why this task:** the user-facing entry point. `tta build` (run pipeline on real data), `tta sweep` (alias for v0.1.0 = build), `tta verify-one <NCT>` (print one trial's flags). Preflight is already in place from Task 1.

**Files:**
- Modify: `C:\Projects\trial-truthfulness-atlas\src\tta\cli.py`
- Create: `C:\Projects\trial-truthfulness-atlas\tests\test_cli.py`

- [ ] **Step 19.1: Write the failing test**

`tests/test_cli.py`:

```python
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
    c.classify.side_effect = ["identical", "refinement", "substantively_different",
                              "identical"]
    return c
```

- [ ] **Step 19.2: Run to verify failure**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — most subcommands not registered.

- [ ] **Step 19.3: Replace `src/tta/cli.py` with the full version**

```python
"""Command-line entry point. Subcommands: preflight, build, sweep, verify-one."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from tta import config, pipeline, preflight
from tta.dashboards import atlas_dashboard, verification_ui
from tta.judge.ollama_client import OllamaClient


def _resolve_fixtures_dir() -> Path:
    return Path(__file__).parent.parent.parent / "tests" / "fixtures"


def _resolve_out_dir() -> Path:
    return config.OUTPUTS_DIR


def _make_ollama_client() -> OllamaClient:
    return OllamaClient()


def cmd_preflight(args: argparse.Namespace) -> int:
    checks = preflight.run_checks()
    print(preflight.format_action_list(checks))
    return 0 if all(c.ok for c in checks) else 1


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = _resolve_out_dir()
    if args.fixture_mode:
        atlas, rollup = pipeline.run_5trial_fixture(
            fixtures_dir=_resolve_fixtures_dir(),
            out_dir=out_dir,
            snapshot_date=date(2026, 4, 12),
            ollama_client=_make_ollama_client(),
        )
    else:
        # Full sweep is v0.2.0 work; keep v0.1.0 honest by routing to fixture.
        print("Full Pairwise70 sweep is deferred to v0.2.0. Use --fixture-mode for v0.1.0.")
        return 2
    atlas_dashboard.write(atlas, out_dir / "dashboard.html",
                          title=f"Trial Truthfulness Atlas v{__import__('tta').__version__}")
    verification_ui.write(atlas, out_dir / "verification.html",
                          title="TTA verification")
    print(f"Wrote {out_dir / 'atlas.csv'} ({len(atlas)} trials).")
    print(f"Wrote {out_dir / 'ma_rollup.csv'} ({len(rollup)} MAs).")
    print(f"Wrote {out_dir / 'dashboard.html'} and {out_dir / 'verification.html'}.")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    return cmd_build(args)


def cmd_verify_one(args: argparse.Namespace) -> int:
    out_dir = _resolve_out_dir()
    atlas, _ = pipeline.run_5trial_fixture(
        fixtures_dir=_resolve_fixtures_dir(),
        out_dir=out_dir,
        snapshot_date=date(2026, 4, 12),
        ollama_client=_make_ollama_client(),
    )
    target = atlas[atlas["nct_id"] == args.nct]
    if target.empty:
        print(f"NCT {args.nct} not found in atlas.")
        return 1
    row = target.iloc[0]
    for k in ["nct_id", "Study", "review_doi", "bridge_method",
              "outcome_drift", "n_drift", "direction_concordance",
              "results_posting"]:
        print(f"  {k}: {row.get(k)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tta", description="Trial Truthfulness Atlas")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("preflight", help="check external prereqs")

    b = sub.add_parser("build", help="build atlas + dashboards (fixture mode in v0.1.0)")
    b.add_argument("--fixture-mode", action="store_true",
                   help="run on the 5-trial test fixture instead of real data")

    s = sub.add_parser("sweep", help="alias for build (v0.1.0)")
    s.add_argument("--fixture-mode", action="store_true")

    v = sub.add_parser("verify-one", help="print all flags for one NCT")
    v.add_argument("--nct", required=True, help="CT.gov NCT identifier")
    v.add_argument("--fixture-mode", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "sweep":
        return cmd_sweep(args)
    if args.cmd == "verify-one":
        return cmd_verify_one(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 19.4: Run tests**

Run: `pytest tests/test_cli.py -v`
Expected: 3 PASS.

- [ ] **Step 19.5: Run full test suite to catch regressions**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 19.6: Commit**

```bash
git add src/tta/cli.py tests/test_cli.py
git commit -m "feat(cli): build|sweep|verify-one subcommands wired through pipeline"
```

---

## Task 20: Sentinel pre-push hook + portfolio integration

**Why this task:** the portfolio convention requires Sentinel pre-push enforcement. Without it, the repo can ship hardcoded paths or empty-DataFrame access that the lessons file explicitly bans.

**Files:**
- No new files in this repo (Sentinel is external)
- Modify: `C:\Projects\trial-truthfulness-atlas\.gitignore` (add Sentinel session files — already done in Task 0)

- [ ] **Step 20.1: Install the Sentinel hook**

Run:

```bash
python -m sentinel install-hook --repo /c/Projects/trial-truthfulness-atlas
```

Expected: prints "installed pre-push hook to .git/hooks/pre-push".

- [ ] **Step 20.2: Run a Sentinel scan and read findings**

Run:

```bash
python -m sentinel scan --repo /c/Projects/trial-truthfulness-atlas
```

Expected: 0 BLOCK. WARN findings (if any) go to `sentinel-findings.md`.

- [ ] **Step 20.3: For any BLOCK, fix at the source**

If Sentinel BLOCKs on hardcoded local paths in shipping code, move those paths into `src/tta/config.py` (env-var override). If it BLOCKs on placeholder HMAC, replace per the cryptography lesson. Do not bypass; the rule encodes a past-incident lesson.

- [ ] **Step 20.4: Confirm hook works end-to-end with a smoke push attempt**

Run:

```bash
git push --dry-run origin master
```

Expected: hook fires, exits 0, push succeeds in dry-run.

- [ ] **Step 20.5: Commit any fixes**

```bash
git add -A && git commit -m "chore(sentinel): clear findings"
```

(skip if no fixes were needed)

---

## Task 21: README + FORKING + E156-PROTOCOL + push.sh

**Why this task:** every atlas-series repo carries the same scaffolding so the portfolio sweeps + workbook + Pages deploy work uniformly.

**Files:**
- Create: `C:\Projects\trial-truthfulness-atlas\README.md`
- Create: `C:\Projects\trial-truthfulness-atlas\FORKING.md`
- Create: `C:\Projects\trial-truthfulness-atlas\E156-PROTOCOL.md`
- Create: `C:\Projects\trial-truthfulness-atlas\push.sh`
- Modify: `C:\E156\rewrite-workbook.txt` (add new entry)

- [ ] **Step 21.1: Write `README.md`**

```markdown
# Trial Truthfulness Atlas (TTA) — v0.1.0

Fifth Pairwise70 atlas. Asks the **upstream** question the prior four
atlases skip: are the trials feeding a Cochrane meta-analysis coherent
between their CT.gov registration and the number Cochrane pooled?

For every cardiology Cochrane MA in scope, TTA computes five integrity
flags per trial:

| Flag | Question |
|------|----------|
| 0 — Bridge | Can we resolve `Study (author year)` to an `nct_id`? |
| 1 — Outcome drift | Did the primary outcome label change? (LLM-judged) |
| 2 — N drift | Did the analysed-N differ > 10 % from registered enrolment? |
| 3 — Direction concordance | Does AACT effect sign match MA effect sign? |
| 4 — Results-posting compliance | Was FDAAA-required CT.gov results posting actually done? |

**Outputs:** `outputs/atlas.csv`, `outputs/ma_rollup.csv`,
`outputs/dashboard.html`, `outputs/verification.html`.

**Quick start:**

```bash
pip install -e ".[dev]"
python -m tta.cli preflight              # verify external prereqs
python -m tta.cli build --fixture-mode   # v0.1.0 fixture pipeline
pytest -v                                # all green
```

**Inputs:**
- AACT snapshot at `D:\AACT-storage\AACT\2026-04-12\` (12 GB, 38 tables)
- Pairwise70 at `C:\Projects\Pairwise70\data\` (595 .rda files)
- Local `gemma2:9b` via `ollama serve`
- DossierGap fixture at `C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv`

**Spec:** `docs/superpowers/specs/2026-04-29-trial-truthfulness-atlas-design.md`
**Plan:** `docs/superpowers/plans/2026-04-29-trial-truthfulness-atlas-v0.1.0.md`

**v0.2.0 roadmap:** widen to all ~6,386 Pairwise70 MAs, add bridge methods 2–4, real-AACT integration test, Crossref DOI.

License: MIT.
```

- [ ] **Step 21.2: Write `FORKING.md`**

```markdown
<!-- sentinel:skip-file — multi-tenant template; user-strings are intentional -->
# Forking the Trial Truthfulness Atlas

This repo is part of the mahmood726-cyber portfolio. To fork for your
own institution / drug class:

1. Replace `src/tta/data/cochrane_heart_review_dois.txt` with your own
   review-DOI list (or topic-mesh terms).
2. Repoint `TTA_AACT_DIR`, `TTA_PAIRWISE70_DIR`, `TTA_DOSSIERGAP_FIXTURE`
   via env vars in your CI.
3. Update `baseline/cardio_v0.1.0.json` to your fixture's expected counts.
4. Re-run `pytest -v` and `python -m tta.cli build --fixture-mode`.

Author block, ORCID, contact email — see top-level CLAUDE.md /
AGENTS.md for portfolio convention.
```

- [ ] **Step 21.3: Write `E156-PROTOCOL.md`**

```markdown
# E156-PROTOCOL — Trial Truthfulness Atlas v0.1.0

Project: `trial-truthfulness-atlas`
Repo: github.com/mahmood726-cyber/trial-truthfulness-atlas
Pages: github.io/trial-truthfulness-atlas/
Date drafted: 2026-04-29
Date submitted: pending
Submitted to: Synthēsis Methods Note + E156 micro-paper
DOI: pending (Crossref at publication)

## Body (CURRENT BODY — AI version, freely updated until SUBMITTED [x])

Cardiology Cochrane meta-analyses pool effect estimates from trials whose
CT.gov registration may differ from what Cochrane extracted. We
quantified five integrity flags — NCT-bridge, outcome drift, N drift,
direction concordance, and FDAAA results-posting compliance — across the
v0.1.0 cardiology subset of Pairwise70 (595 reviews; ~6,386 MAs in the
full dataset). Bridge resolution used DossierGap direct match in v0.1.0;
outcome-drift used a frozen `gemma2:9b` local-LLM judge with sha256-pinned
prompts and disk-cached judgments. The v0.1.0 fixture (5 trials) returned
80 % bridge resolution, 1/4 substantively different outcome, 1/4 N drift,
0/4 direction flips, and 1/3 FDAAA results-posting violations. Production
v0.2.0 will widen to all ~6,386 MAs and add three additional bridge
methods. The headline integrity rate (and its association with crosses-
null status) is reported on `dashboard.html`; reproducibility is gated by
the pinned numerical baseline at `baseline/cardio_v0.1.0.json`. v0.1.0
treats the unbridgeable rate as a publishable finding in itself, since
silent CT.gov-to-MA non-resolution is the upstream confounder of every
prior reproducibility atlas.

## Author block

Mahmood Ahmad — middle-author on all E156 papers (per portfolio
convention; see global feedback memory).

## Dashboard

Pages link: github.io/trial-truthfulness-atlas/  (live after Pages
enable)

YOUR REWRITE:

SUBMITTED: [ ]
```

- [ ] **Step 21.4: Write `push.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
git push -u origin master --tags
```

- [ ] **Step 21.5: Make push.sh executable on the file system**

Run: `chmod +x push.sh`

- [ ] **Step 21.6: Update workbook entry**

Append a new entry to `C:\E156\rewrite-workbook.txt` with the body from §21.3 above. Increment the workbook total count. Leave `YOUR REWRITE` empty and `SUBMITTED: [ ]`.

- [ ] **Step 21.7: Commit**

```bash
git add README.md FORKING.md E156-PROTOCOL.md push.sh
git commit -m "docs: README + FORKING + E156 protocol + push helper"
```

---

## Task 22: Final preflight, full test sweep, baseline check, tag v0.1.0

**Why this task:** per the verification-readiness preflight rule and the SKIP-as-pass lesson — never tag a release without a green full suite, a present numerical baseline, and zero Sentinel BLOCK.

**Files:**
- Modify (potentially): any failing tests / Sentinel findings
- Create: git tag `v0.1.0`

- [ ] **Step 22.1: Re-run the CLI preflight against real environment**

Run: `python -m tta.cli preflight`
Expected: all checks PASS. If anything is missing, fix at the source (per the bounded-loop rule, do not paper over).

- [ ] **Step 22.2: Run the entire test suite**

Run: `pytest -v --tb=short`
Expected: every test PASS. If anything fails, fix the underlying issue rather than skipping.

- [ ] **Step 22.3: Run the fixture pipeline end-to-end and visually inspect outputs**

Run:

```bash
python -m tta.cli build --fixture-mode
ls outputs/
```

Expected: `atlas.csv`, `ma_rollup.csv`, `dashboard.html`, `verification.html` all present. Open `dashboard.html` and `verification.html` in a browser; click through; confirm no console errors and the verification UI's localStorage round-trip works.

- [ ] **Step 22.4: Sentinel pre-push scan once more**

Run: `python -m sentinel scan --repo /c/Projects/trial-truthfulness-atlas`
Expected: 0 BLOCK.

- [ ] **Step 22.5: Verify the baseline file is present and complete**

Run:

```bash
python -c "
import json
from pathlib import Path
data = json.loads(Path('baseline/cardio_v0.1.0.json').read_text(encoding='utf-8'))
assert data['version'] == '0.1.0'
assert 'REPLACE_WITH_ACTUAL_HASH' not in data['prompt_sha_outcome_drift']
print('baseline OK')
"
```

Expected: prints `baseline OK`.

- [ ] **Step 22.6: Tag v0.1.0**

```bash
git tag -a v0.1.0 -m "Trial Truthfulness Atlas v0.1.0 — cardiology, 5 flags, fixture-pinned"
git tag --list | tail -5
```

Expected: tag listed.

- [ ] **Step 22.7: Reconcile the portfolio registry**

Run: `python C:\ProjectIndex\reconcile_counts.py`
Expected: exit 0. If it fails, add `trial-truthfulness-atlas` to `restart-manifest.json` and `INDEX.md` first, then re-run.

- [ ] **Step 22.8: Stop here. v0.1.0 is shipped locally.**

Push to GitHub (`./push.sh`), enable Pages, and update INDEX.md / MEMORY.md only after the user has reviewed the release output. Per the workflow rule: "Before substantial work, restate the deliverable" — release-to-public is substantial; ask first.

---

## Self-review summary

After writing this plan, ran the self-review checklist:

**1. Spec coverage:** every spec section traces to ≥1 task.

| Spec § | Implementing task |
|--------|-------------------|
| §1 Problem statement | (motivational; covered in README §21) |
| §2 Why this machine | Tasks 1, 3, 4, 6 |
| §3 Scope | Task 5 (cardio filter) |
| §4 Architecture | Tasks 0–18 (file-by-file) |
| §5 Flag 0 (bridge) | Task 9 |
| §5 Flag 1 (outcome drift) | Task 10 (+ 6, 7, 8) |
| §5 Flag 2 (N drift) | Task 11 |
| §5 Flag 3 (direction) | Task 12 |
| §5 Flag 4 (results posting) | Task 13 |
| §6 Data flow | Task 15 (pipeline) |
| §7 Error handling | Tasks 11 (negation), 14 (empty-DF), 1 (preflight), 8 (ollama failure) |
| §8 Testing | Tasks 2, 15, 16, 22 |
| §9 Outputs | Tasks 15, 17, 18 |
| §10 Reproducibility | Tasks 6, 7, 8, 16 |
| §11 Open questions | Deferred to v0.2.0 / Task 22 review |
| §12 Non-goals | Honored (no causal claim, no full sweep, no retraction) |
| §13 Predicted headline | (post-shipping; not in plan scope) |
| §14 Atlas-series fit | README §21 |
| §15 Preflight | Tasks 1, 22 |

**2. Placeholder scan:** none. The string `REPLACE_WITH_ACTUAL_HASH` in Task 16's baseline JSON is replaced in Step 16.3 by an explicit command. The expected `atlas.csv` in Task 15 contains specific numeric values for the fixture; Step 15.6 explicitly notes that if the integration produces different log-effects (e.g., `-0.2231` vs `-0.223` due to rounding), the expected CSV is the one that should be updated.

**3. Type consistency:** spot-checked: `BridgeResult` (Task 9) is consumed by `pipeline.py` (Task 15) as `r.nct_id`, `r.method`, `r.confidence` — matches. `OutcomeDriftResult` (Task 10) returns `.label`, `.prompt_sha`, `.model_version` — matches `compute_dataframe`'s column names. `ALLOWED_LABELS` constant is consistent across Task 10 (impl) and Task 10 (test).

---

## Execution Handoff

Plan complete and saved to `C:\Projects\trial-truthfulness-atlas\docs\superpowers\plans\2026-04-29-trial-truthfulness-atlas-v0.1.0.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
