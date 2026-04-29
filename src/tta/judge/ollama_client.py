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
