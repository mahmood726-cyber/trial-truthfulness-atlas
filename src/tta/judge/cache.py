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
