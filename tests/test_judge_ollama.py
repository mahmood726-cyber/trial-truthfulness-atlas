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
    assert client.get_model_version() == "gemma2:9b@ff02c3702f32"


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
