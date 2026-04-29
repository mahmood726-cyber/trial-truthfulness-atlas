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
