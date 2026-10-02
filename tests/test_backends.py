"""No-model tests for the backends: path resolution, cue mapping, and the ollama
client's parsing. Nothing here loads a model or touches the network."""

import io
import json
import sys
import types

import numpy as np

import backends
from backends import WhisperASR, resolve_model


# ------------------------------------------------------------------ resolve


def test_resolve_model_prefers_existing_local_dir(tmp_path):
    assert resolve_model(str(tmp_path)) == str(tmp_path)


def test_resolve_model_passes_through_repo_ids():
    assert resolve_model("mlx-community/whisper-large-v3-mlx") == \
        "mlx-community/whisper-large-v3-mlx"


# ------------------------------------------------------------------ backend


def test_backend_auto_detects_and_translates():
    asr = backends.get_backend()
    assert isinstance(asr, WhisperASR)
    assert asr.language is None  # None -> whisper auto-detects
    assert asr.task == "translate"
    assert asr.model == backends.TRANSLATE_ASR


def test_whisper_maps_segments_with_offset_and_skips_silence(monkeypatch):
    fake = types.ModuleType("mlx_whisper")
    fake.transcribe = lambda audio, **kw: {
        "segments": [
            {"start": 0.5, "end": 1.5, "text": " hello ", "no_speech_prob": 0.1},
            {"start": 1.5, "end": 2.0, "text": " hmm ", "no_speech_prob": 0.9},
            {"start": 2.0, "end": 2.5, "text": "   ", "no_speech_prob": 0.1},
        ]
    }
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    cues = WhisperASR("whatever").run(np.zeros(16000, dtype=np.float32), 10.0)
    assert [(c.start, c.end, c.source) for c in cues] == [(10.5, 11.5, "hello")]


# ----------------------------------------------------------------- ollama


class _FakeHTTP:
    """Stands in for urlopen so the ollama client can be tested with no server.

    This is the only boundary that talks to something other than Whisper, and it
    was the one client with no coverage at all.
    """

    def __init__(self, tags=None, content=None, boom=False):
        self.tags = tags if tags is not None else []
        self.content = content
        self.boom = boom
        self.posted = []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if self.boom:
            raise OSError("connection refused")
        if url.endswith("/api/tags"):
            body = {"models": [{"name": n} for n in self.tags]}
        else:
            self.posted.append(req)
            body = {"message": {"content": self.content}}
        return io.BytesIO(json.dumps(body).encode())


def _llm(monkeypatch, http, **kw):
    monkeypatch.setattr(backends.urllib.request, "urlopen", http)
    return backends.OllamaLLM(**kw)


def test_has_model_accepts_both_the_bare_and_latest_tag(monkeypatch):
    # ollama reports `qwen3:8b` when you ask for `qwen3:8b:latest` and vice
    # versa, so either spelling has to satisfy the check.
    assert _llm(monkeypatch, _FakeHTTP(tags=["qwen3:8b"])).has_model() is True
    assert _llm(monkeypatch, _FakeHTTP(tags=["qwen3:8b:latest"])).has_model() is True
    assert _llm(monkeypatch, _FakeHTTP(tags=["llama3"])).has_model() is False


def test_has_model_is_false_when_ollama_is_not_running(monkeypatch):
    """The whole point of the check: stay off rather than let /api/chat pull a
    multi-GB model nobody asked for."""
    assert _llm(monkeypatch, _FakeHTTP(boom=True)).has_model() is False


def test_chapters_parse_sort_and_drop_garbage(monkeypatch):
    reply = json.dumps({"chapters": [
        {"start": 300, "title": "Second"},
        {"start": 0, "title": '"First, part one."'},
        {"start": 150, "title": "Middle"},
        {"start": "not a number", "title": "Bad start"},
        {"start": 200, "title": "   "},
        {"start": 250, "title": None},
        {"start": True, "title": "Boolean start"},  # bool is an int subclass
    ]})
    llm = _llm(monkeypatch, _FakeHTTP(content=reply))
    assert llm.chapters("transcript") == [
        {"start": 0.0, "title": "First, part one"},
        {"start": 150.0, "title": "Middle"},
        {"start": 300.0, "title": "Second"},
    ]


def test_chapters_clamp_negative_starts_and_truncate_to_the_cap(monkeypatch):
    reply = json.dumps({"chapters": [
        {"start": -50, "title": "Clamped"},
        *[{"start": i * 10, "title": f"T{i}"} for i in range(10)],
    ]})
    llm = _llm(monkeypatch, _FakeHTTP(content=reply), max_chapters=3)
    out = llm.chapters("transcript")
    assert len(out) == 3, "the cap must hold even when the model overshoots"
    assert out[0]["start"] == 0.0, "a negative start is meaningless as an offset"


def test_a_reply_with_no_chapters_is_not_an_error(monkeypatch):
    for reply in ('{"chapters": null}', '{"chapters": []}', "{}"):
        assert _llm(monkeypatch, _FakeHTTP(content=reply)).chapters("t") == []


def test_chapter_titles_are_capped_at_eighty_chars(monkeypatch):
    reply = json.dumps({"chapters": [{"start": 0, "title": "x" * 200}]})
    assert len(_llm(monkeypatch, _FakeHTTP(content=reply)).chapters("t")[0]["title"]) == 80


def test_the_request_asks_ollama_not_to_think_or_pull(monkeypatch):
    """`think: false` stops qwen3 burning its budget on reasoning instead of
    titles, and `keep_alive` evicts the model instead of parking it in unified
    memory beside Whisper."""
    http = _FakeHTTP(content='{"chapters": []}')
    _llm(monkeypatch, http).chapters("transcript", prev="Earlier")
    sent = json.loads(http.posted[0].data)
    assert sent["think"] is False
    assert sent["stream"] is False
    assert sent["keep_alive"] == "10m"
    assert sent["format"] == backends.CHAPTER_SCHEMA
    assert sent["options"]["temperature"] == 0.4
    # The previous title travels with the prompt so windows don't repeat.
    user = sent["messages"][1]["content"]
    assert "Earlier" in user and "(none)" not in user
