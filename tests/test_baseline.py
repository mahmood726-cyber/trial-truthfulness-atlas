from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path(__file__).parent.parent / "baseline" / "cardio_v0.1.0.json"


def test_baseline_file_exists():
    assert BASELINE.is_file()


def test_baseline_required_keys():
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["version"] in {"0.1.0", "0.1.1"}
    assert data["snapshot_date"] == "2026-04-12"
    # Fixture: 3 .rda files in v0.1.1 (was 2 in v0.1.0; GRIPHON moved to its
    # own .rda because no Cochrane review pools PAH and HFrEF). Real-data
    # sweep would have 590+.
    assert data["pairwise70_files"] >= 590 or data["pairwise70_files"] in {2, 3}
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
    assert "REPLACE" not in data["prompt_sha_outcome_drift"]
