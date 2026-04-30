"""Flag 1 — semantic outcome drift between CT.gov registration and MA-extracted outcome."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from tta.judge import cache as cache_mod
from tta.judge import prompts
from tta.judge.ollama_client import OllamaClient

ALLOWED_LABELS = {"identical", "refinement", "substantively_different"}

# V2 splits substantively_different into two more specific labels.
ALLOWED_LABELS_V2 = {"identical", "refinement", "construct_change", "time_point_shift"}


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
    # Guard against both Python None and pandas float NaN (truthy in Python).
    if pd.isnull(registered_outcome) or not registered_outcome or pd.isnull(ma_extracted_outcome) or not ma_extracted_outcome:
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
