"""Local ollama HTTP wrapper with deterministic seed and label gating."""

from __future__ import annotations

from typing import Set
from urllib.parse import urlparse

import requests

from tta import config

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


class RemoteOllamaError(RuntimeError):
    """Raised when OllamaClient is asked to call a non-loopback URL without
    the TTA_ALLOW_REMOTE_OLLAMA opt-in. Prevents accidental data exfiltration
    of trial outcomes via a hostile env var."""


def _assert_loopback_or_opted_in(url: str) -> None:
    if config.ALLOW_REMOTE_OLLAMA:
        return
    host = urlparse(url).hostname or ""
    if host.lower() not in _LOOPBACK_HOSTS:
        raise RemoteOllamaError(
            f"refusing to call non-loopback ollama URL ({url!r}). "
            "Set TTA_ALLOW_REMOTE_OLLAMA=1 if this is intentional."
        )


class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None, timeout: int = 60):
        self.url = url or config.OLLAMA_URL
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout
        _assert_loopback_or_opted_in(self.url)

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
