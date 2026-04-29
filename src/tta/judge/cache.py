"""sha256-keyed JSON disk cache for LLM judgments."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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

    def purge_other_versions(self, keep_model_version: str) -> int:
        """Delete cache entries whose `model_version` payload field does not
        match `keep_model_version`. Returns the count removed.

        Use after upgrading the LLM model to reclaim disk space — a full
        Pairwise70 sweep at v0.2.0 scale produces thousands of entries; old
        model digests stay valid in the on-disk cache forever otherwise.
        Entries that fail to parse are also removed (corrupt cache files
        cannot satisfy a future hit anyway).
        """
        removed = 0
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("removing unreadable cache entry %s", path)
                path.unlink(missing_ok=True)
                removed += 1
                continue
            if data.get("model_version") != keep_model_version:
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            logger.info("purged %d cache entries for non-matching model versions", removed)
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))
