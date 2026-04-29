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


def test_len_counts_cached_entries(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    assert len(c) == 0
    c.put("a", {"label": "identical"})
    c.put("b", {"label": "refinement"})
    assert len(c) == 2


def test_purge_other_versions_removes_stale_model_entries(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    c.put("k_old1", {"label": "identical", "model_version": "gemma2:9b@old"})
    c.put("k_old2", {"label": "refinement", "model_version": "gemma2:9b@old"})
    c.put("k_new", {"label": "concordant", "model_version": "gemma2:9b@new"})
    removed = c.purge_other_versions("gemma2:9b@new")
    assert removed == 2
    assert c.get("k_old1") is None
    assert c.get("k_old2") is None
    assert c.get("k_new") == {"label": "concordant", "model_version": "gemma2:9b@new"}


def test_purge_removes_unreadable_entries(judge_cache_dir):
    c = cache.JudgeCache(judge_cache_dir)
    c.put("ok", {"label": "identical", "model_version": "v1"})
    # Manually corrupt one entry
    (judge_cache_dir / "corrupt.json").write_text("not valid json {{", encoding="utf-8")
    removed = c.purge_other_versions("v1")
    assert removed == 1  # only the corrupt one was purged; "ok" matches v1
    assert c.get("ok") is not None
