"""No-model tests for the backends: path resolution, cue mapping, and the mlx-lm
chapter client's parsing. Nothing here loads weights or touches the network."""

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


# ------------------------------------------------------------------- chapters


class _FakeGenerate:
    """Stands in for mlx_lm.generate so the chapter client is tested with no
    weights. Returns a canned reply and records the prompt it was handed.
    """

    def __init__(self, reply=""):
        self.reply = reply
        self.kwargs = None
        self.prompt = None

    def __call__(self, model, tokenizer, prompt, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        return self.reply

    def as_generate(self, monkeypatch):
        """A tokenizer whose only job is to hand back a recognisable prompt."""
        monkeypatch.setattr(backends, "generate", self)
        monkeypatch.setattr(backends, "load", lambda repo: (object(), _FakeTokenizer()))
        return self

    @property
    def user(self):
        return self.prompt.split("<user>")[-1]


class _FakeTokenizer:
    has_thinking = False

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(
            f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages)


def _llm(monkeypatch, reply="", **kw):
    gen = _FakeGenerate(reply).as_generate(monkeypatch)
    return backends.MLXLLM(**kw), gen


def test_has_model_is_true_for_a_local_dir(tmp_path):
    # A weights dir on disk needs no cache probe at all.
    assert backends.MLXLLM(model=str(tmp_path)).has_model() is True


def test_has_model_is_false_when_the_repo_is_not_cached(monkeypatch):
    """The whole point of the check: stay off rather than let the first window
    pull a couple of GB nobody asked for."""
    def boom(*a, **kw):
        raise OSError("not in the cache")
    monkeypatch.setattr(backends, "snapshot_download", boom)
    assert backends.MLXLLM(model="mlx-community/nope-4bit").has_model() is False


def test_has_model_is_true_when_the_weights_are_cached(monkeypatch):
    seen = {}

    def cached(repo, **kw):
        seen.update(repo=repo, **kw)
        return "/cache/snapshot"
    monkeypatch.setattr(backends, "snapshot_download", cached)
    assert backends.MLXLLM(model="mlx-community/x-4bit").has_model() is True
    # Must never reach the network, or this check stalls every websocket hello.
    assert seen["local_files_only"] is True
    # ...and must ask about the weights, not any file: a cache holding only the
    # tokenizer would otherwise read as ready and cost 2 GB on the first window.
    assert seen["allow_patterns"] == ["*.safetensors"]


def test_chapters_parse_sort_and_drop_garbage(monkeypatch):
    llm, _ = _llm(monkeypatch,
                  "00:00 - \"First, part one.\"\n"
                  "02:30 - Second\n"
                  "01:30 - Middle\n"
                  "04:00 - \n"           # no title: dropped
                  "  \n"                 # not a chapter line at all
                  "05:00 - Fine")
    assert llm.chapters("transcript") == [
        {"start": 0.0, "title": "First, part one"},
        {"start": 90.0, "title": "Middle"},
        {"start": 150.0, "title": "Second"},
        {"start": 300.0, "title": "Fine"},
    ]


def test_chapters_ignore_transcript_lines_quoted_back_by_the_model(monkeypatch):
    """The transcript's own `[mm:ss] text` lines must not read as chapters: they
    are bracketed and carry no dash, so the pattern cannot match them."""
    llm, _ = _llm(monkeypatch,
                  "[01:30] and that is why the buffer fills\n"
                  "00:00 - The real topic change\n"
                  "02:00 - The next one")
    assert [c["title"] for c in llm.chapters("transcript")] == [
        "The real topic change", "The next one"]


def test_chapters_clamp_to_the_cap(monkeypatch):
    llm, _ = _llm(monkeypatch,
                  "".join(f"00:{i * 10:02d} - T{i}\n" for i in range(10)),
                  max_chapters=3)
    out = llm.chapters("transcript")
    assert len(out) == 3, "the cap must hold even when the model overshoots"


def test_a_reply_with_no_chapters_is_not_an_error(monkeypatch):
    # Chapterer falls back to "Continued" itself, so a flat refusal is not fatal.
    for reply in ("I cannot help with that.", "", "```\n00:00 - Kept\n```"):
        llm, _ = _llm(monkeypatch, reply)
        out = llm.chapters("t")
        assert [c["title"] for c in out] in ([], ["Kept"])


def test_chapter_titles_are_capped_at_eighty_chars(monkeypatch):
    llm, _ = _llm(monkeypatch, "00:00 - " + "x" * 200)
    assert len(llm.chapters("t")[0]["title"]) == 80


def test_truncation_keeps_the_chapters_that_were_finished(monkeypatch):
    """The reason this is a line format and not JSON: a reply cut off by
    max_tokens still yields the lines it completed. A truncated JSON reply throws
    away every chapter in it and the whole window is retried."""
    llm, _ = _llm(monkeypatch,
                  "00:00 - First\n01:47 - Second\n02:59 - Third half writ")
    assert [c["title"] for c in llm.chapters("t")] == [
        "First", "Second", "Third half writ"]


def test_a_titleless_line_does_not_swallow_the_next_chapter(monkeypatch):
    """`\\s` matches newlines, so a permissive pattern reads an empty title as
    the following line and welds two chapters into one."""
    llm, _ = _llm(monkeypatch, "04:00 -   \n05:00 - Kept")
    assert llm.chapters("t") == [{"start": 300.0, "title": "Kept"}]


def test_the_previous_title_travels_with_the_prompt(monkeypatch):
    # So adjacent windows don't repeat a chapter title.
    llm, gen = _llm(monkeypatch, "00:00 - Next")
    llm.chapters("transcript", prev="Earlier")
    assert "Earlier" in gen.user and "(none)" not in gen.user

    llm, gen = _llm(monkeypatch, "00:00 - Next")
    llm.chapters("transcript")
    assert "(none)" in gen.user


def test_generation_is_bounded_and_leaves_reasoning_off(monkeypatch):
    """max_tokens bounds the worker thread (there is no request timeout to lean
    on), and thinking off keeps a reasoning model from spending its budget on
    titles instead of writing them."""
    llm, gen = _llm(monkeypatch, "00:00 - Next", max_tokens=64)
    llm.chapters("t")
    assert gen.kwargs["max_tokens"] == 64


def test_thinking_is_disabled_for_models_that_have_it(monkeypatch):
    gen = _FakeGenerate("00:00 - Next").as_generate(monkeypatch)

    class Thinking(_FakeTokenizer):
        has_thinking = True
        seen = None

        def apply_chat_template(self, messages, **kwargs):
            type(self).seen = kwargs
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(backends, "load", lambda repo: (object(), Thinking()))
    backends.MLXLLM().chapters("t")
    assert Thinking.seen["enable_thinking"] is False
