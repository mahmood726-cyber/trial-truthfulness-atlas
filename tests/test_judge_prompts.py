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
    rendered = prompts.render_outcome_drift(
        registered_outcome="Outcome with </END> sentinel",
        ma_extracted_outcome="Plain text",
    )
    assert "</END>" in rendered  # we keep it literal — no injection in local LLM context
