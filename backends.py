"""Whisper backend: auto-detect the spoken language, translate to English.

Whisper's built-in `task="translate"` does any-language -> English in one pass,
so there is a single backend and no separate translation stage.

Kokoro speaks those translated cues back for read-along. It needs espeak-ng as
its grapheme-to-phoneme step, like every small open English TTS.

mlx-lm names a topic every so often so the panel gets a chapter list. It runs
in this process on the chapter worker thread, so its weights load on the first
window rather than at boot and then stay resident next to Whisper.

Bark 3.0 Clean is the third engine: an English-only Parakeet fine-tune that
drops fillers, so the cues read clean with no second pass. Its weights come as
one entropy-coded blob; bark_weights.py unpacks them on first use.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from pipeline import CACHE_DIR, SAMPLE_RATE, Cue

# large-v3-turbo silently ignores task="translate" (it just transcribes), so use
# the full large-v3 weights, which are translate-capable.
TRANSLATE_ASR = os.environ.get("LT_TRANSLATE_MODEL", "mlx-community/whisper-large-v3-mlx")

# Two engines behind one interface. Whisper auto-detects any language and
# translates to English; Phonon-2 is English-only and ~10x faster at 164 MB.
ASR = os.environ.get("LT_ASR", "phonon").strip().lower()
PHONON_MODEL = os.environ.get("LT_PHONON_MODEL", "phonon-2")
ASR_BACKENDS = ("whisper", "phonon", "bark")

# Bark 3.0 Clean: English-only, and it deletes "um", false starts and repeats
# on purpose. Weights arrive as a 180 MB .bark blob that unpacks to ~1.2 GB of
# mlx-audio safetensors, so it costs one slow first use. Optional at runtime:
# nothing here loads it unless the engine is selected.
BARK_MODEL = os.environ.get("LT_BARK_MODEL", "ConwayResearch/Underdog-Bark-3.0-Clean")
# The vendored decoder reads their private format, so load the revision it was
# written against; bump only after re-checking an unpack against this one.
BARK_REVISION = os.environ.get(
    "LT_BARK_REVISION", "6d34ed8639b2b9a33e764acf06748519bbb950cf")

# Kokoro is the fastest of the good small English voices and measured 12.7x
# realtime on an M1 Pro, so a cue is spoken in ~4% of the look-ahead window.
# 4bit is indistinguishable from bf16 in speed and ~20% lighter.
TTS_MODEL = os.environ.get("LT_TTS_MODEL", "mlx-community/Kokoro-82M-4bit")
TTS_VOICE = os.environ.get("LT_TTS_VOICE", "af_heart")
TTS_LANG = os.environ.get("LT_TTS_LANG", "a")  # 'a' = American English

# mlx-lm names the chapter titles. A small instruct model is enough: the only job
# is naming a topic from ~1.5k words of context, so the thing that matters is
# finding topic boundaries reliably, not fluency.
#
# Measured on a three-topic window, 6 runs each, both found all 3 chapters:
# Qwen3.5-4B found both real topic changes 6/6, Llama-3.2-3B only 4/6. For 1.2 GB
# and 1.2s more, against a 300s window where 1.2s is nothing.
CHAPTER_MODEL = os.environ.get("LT_CHAPTER_MODEL",
                               "mlx-community/Qwen3.5-4B-MLX-4bit")
CHAPTER_MAX_PER_WINDOW = 4


def _point_at_espeak() -> None:
    """Teach phonemizer where Homebrew put espeak-ng.

    ctypes.util.find_library doesn't search Homebrew prefixes, so phonemizer
    reports "espeak not installed" on a perfectly good install. Both vars are
    phonemizer's own documented overrides, so nothing here is phonemizer-specific.
    """
    for prefix in ("/opt/homebrew", "/usr/local"):  # Apple Silicon, then Intel
        lib = Path(prefix) / "lib" / "libespeak-ng.dylib"
        data = Path(prefix) / "share" / "espeak-ng-data"
        if lib.is_file() and data.is_dir():
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(lib))
            os.environ.setdefault("PHONEMIZER_ESPEAK_DATA_PATH", str(data))
            return


_point_at_espeak()


def hf_token() -> str | None:
    """HF_TOKEN (or the legacy HUGGING_FACE_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def resolve_model(repo_or_path: str) -> str:
    """A local dir (e.g. an HF snapshot) is used as-is; otherwise the repo id.

    mlx-whisper (`path_or_hf_repo`) accepts either, so anything already sitting
    in the HF cache on this machine works offline with no download.
    """
    p = Path(os.path.expanduser(repo_or_path))
    return str(p) if p.is_dir() else repo_or_path


# mlx-whisper calls huggingface_hub internally and reads the token from the
# environment, so make sure a legacy var is visible under the standard name.
if hf_token():
    os.environ.setdefault("HF_TOKEN", hf_token())


class WhisperASR:
    """Whisper via MLX. `task="translate"` is Whisper's built-in X->English."""

    def __init__(self, model: str = TRANSLATE_ASR, language: str | None = None,
                 task: str = "translate"):
        self.model = model
        self.language = language
        self.task = task
        self._prompt: str | None = None
        # mlx_whisper's model cache is process-global, so serialise the first
        # load: a startup warm and an early chunk must not both load it.
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Load the weights now so the first chunk doesn't stall mid-playback."""
        import mlx_whisper

        with self._lock:
            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=resolve_model(self.model),
                language=self.language,
                task=self.task,
                condition_on_previous_text=False,
            )

    def run(self, audio, offset: float) -> list[Cue]:
        import mlx_whisper

        with self._lock:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=resolve_model(self.model),
                language=self.language,
                task=self.task,
                initial_prompt=self._prompt,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )

        cues: list[Cue] = []
        for seg in result.get("segments", []):
            if (seg.get("no_speech_prob") or 0.0) > 0.6:
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            cues.append(Cue(offset + seg["start"], offset + seg["end"], text))

        if cues:
            tail = " ".join(c.source for c in cues[-8:])
            self._prompt = tail[-200:]
        return cues


class PhononASR:
    """English-only transcription via fermion (Phonon-2, MLX).

    Same interface as WhisperASR: run(audio, offset) -> list[Cue] with
    absolute times. Phonon segments already arrive sentence-sized with their
    own timings, so mapping is offset + filter.
    """

    def __init__(self, model: str = PHONON_MODEL):
        self.model = model
        self._model = None
        # fermion holds its own resident model, so serialise callers the same
        # way: warm + first chunk must not load twice.
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            try:
                import fermion
            except ImportError as exc:
                raise RuntimeError(
                    "phonon needs the fermion-research package "
                    "(`pip install fermion-research`)") from exc
            self._model = fermion.load_speech(self.model)
        return self._model

    def warm(self) -> None:
        """Load the weights now so the first chunk doesn't stall mid-playback."""
        with self._lock:
            self._load().transcribe_array(
                np.zeros(SAMPLE_RATE, dtype=np.float32))

    def run(self, audio, offset: float) -> list[Cue]:
        with self._lock:
            tr = self._load().transcribe_array_detailed(
                np.asarray(audio, dtype=np.float32))

        cues: list[Cue] = []
        for seg in tr.segments or []:
            if isinstance(seg, dict):
                text, start, end = seg.get("text", ""), seg.get("start", 0.0), seg.get("end", 0.0)
            else:
                text, start, end = seg.text, seg.start, seg.end
            if (text or "").strip() and end > start:
                cues.append(Cue(offset + start, offset + end, text.strip()))
        if not cues:
            # Untimed fallback: text without segments is still transcript.
            text = (tr.text or "").strip()
            if text:
                cues.append(Cue(offset, offset + len(audio) / SAMPLE_RATE, text))
        return cues


def _bark_folder(model: str = BARK_MODEL) -> Path:
    """The unpacked mlx-audio folder for Bark; downloads and unpacks once."""
    out = CACHE_DIR / "models" / f"bark-{BARK_REVISION[:8]}"
    if (out / "model.safetensors").is_file():
        return out
    local = Path(os.path.expanduser(model))
    repo = local if local.is_dir() else Path(snapshot_download(
        model, revision=BARK_REVISION,
        allow_patterns=["model.bark", "mlx/*", "NOTICE"], token=hf_token()))
    from bark_weights import prepare

    meta = repo / "mlx"
    return prepare(repo / "model.bark", meta if meta.is_dir() else repo, out)


class BarkASR:
    """English-only Parakeet fine-tune that removes fillers (Bark 3.0 Clean).

    Same interface as the others: run(audio, offset) -> list[Cue] with
    absolute times. mlx-audio returns sentence bounds with word timings, so
    mapping is offset + filter, and text-only output still produces a cue.
    """

    def __init__(self, model: str = BARK_MODEL):
        self.model = model
        self._model = None
        # One resident model, first load serialised like the others.
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import mlx_audio.stt

            self._model = mlx_audio.stt.load(_bark_folder(self.model),
                                             model_type="parakeet")
        return self._model

    def warm(self) -> None:
        """Load the weights now so the first chunk doesn't stall mid-playback."""
        import mlx.core as mx

        with self._lock:
            self._load().generate(mx.zeros(SAMPLE_RATE), dtype=mx.bfloat16,
                                  chunk_duration=None, verbose=False)

    def run(self, audio, offset: float) -> list[Cue]:
        import mlx.core as mx

        with self._lock:
            res = self._load().generate(mx.array(np.asarray(audio, np.float32)),
                                        dtype=mx.bfloat16,
                                        chunk_duration=None, verbose=False)

        cues: list[Cue] = []
        for s in getattr(res, "sentences", None) or []:
            text = (s.text or "").strip()
            if text and s.end > s.start:
                cues.append(Cue(offset + s.start, offset + s.end, text))
        if not cues:
            text = (getattr(res, "text", "") or "").strip()
            if text:
                cues.append(Cue(offset, offset + len(audio) / SAMPLE_RATE, text))
        return cues


_backends: dict = {}
_asr_name = ASR if ASR in ASR_BACKENDS else "whisper"


def current_backend_name() -> str:
    return _asr_name


def set_backend(name: str):
    """Switch engine. Each stays resident once loaded. Takes effect on the next
    open — the client reopens the current video on switch, so in practice the
    whole video is retranscribed with the new engine."""
    global _asr_name
    name = (name or "").strip().lower()
    if name not in ASR_BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r} (want one of {ASR_BACKENDS})")
    _asr_name = name
    return get_backend()


def get_backend(name: str | None = None):
    """The resident model. No arg = the currently selected engine."""
    if name is None:
        name = _asr_name
    if name not in ASR_BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r} (want one of {ASR_BACKENDS})")
    if name not in _backends:
        if name == "whisper":
            _backends[name] = WhisperASR(TRANSLATE_ASR, None, "translate")
        elif name == "phonon":
            _backends[name] = PhononASR()
        else:
            _backends[name] = BarkASR()
    return _backends[name]


def asr_state() -> dict:
    return {"current": _asr_name, "available": list(ASR_BACKENDS),
            "models": {"whisper": TRANSLATE_ASR, "phonon": PHONON_MODEL,
                       "bark": BARK_MODEL}}


# --------------------------------------------------------------------------
# Kokoro: speaks the translated cues for read-along
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def speech_text(raw: str) -> str:
    """Cue text -> something worth speaking.

    reflow_cues emits two-line cues and Kokoro's default split_pattern is r"\\n+",
    so the newline is folded into a space rather than splitting one cue into two
    prosody resets. Whisper's occasional [Music] / ♪ noise is dropped so it never
    reaches the synthesiser.
    """
    text = _WS.sub(" ", raw or "").strip()
    text = re.sub(r"[\[\(](music|applause|inaudible|silence)[^\]\)]*[\]\)]", " ", text, flags=re.I)
    text = text.replace("♪", " ").replace("♫", " ")
    return _WS.sub(" ", text).strip(" -–—")


class KokoroTTS:
    """Kokoro via MLX. Loaded on first use, not at boot."""

    sample_rate = 24000

    # The published Kokoro-82M voice set. Hardcoded so the UI can render a
    # picker without a round-trip that downloads the voices directory.
    voices = (
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova",
        "af_aoede", "af_alloy", "af_jessica", "af_kore", "af_river",
        "am_michael", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_onyx", "am_puck", "am_santa",
        "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
        "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
    )

    def __init__(self, model: str = TTS_MODEL, voice: str = TTS_VOICE, lang: str = TTS_LANG):
        self.model = model
        self.voice = voice
        self.lang = lang
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(resolve_model(self.model))
        return self._model

    def warm(self) -> None:
        """Pay the model + phonemizer load now, off the caller's critical path."""
        import mlx.core as mx

        with self._lock:
            for _ in self._load().generate(text="ready.", voice=self.voice, lang_code=self.lang):
                pass
            mx.clear_cache()

    def speak(self, text: str, speed: float = 1.0) -> np.ndarray | None:
        """Synthesise one cue. Returns mono float32 at `sample_rate`, or None if
        there was nothing speakable in the text."""
        import mlx.core as mx

        text = speech_text(text)
        if not text:
            return None

        with self._lock:
            chunks = [
                np.asarray(r.audio, dtype=np.float32)
                for r in self._load().generate(
                    text=text, voice=self.voice, speed=speed, lang_code=self.lang
                )
            ]
            # Kokoro compiles its decoder graph; without this the cache grows
            # unbounded next to a resident Whisper model.
            mx.clear_cache()

        if not chunks:
            return None
        audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return audio if audio.size else None


_tts: KokoroTTS | None = None


def get_tts() -> KokoroTTS:
    global _tts
    if _tts is None:
        _tts = KokoroTTS()
    return _tts


# --------------------------------------------------------------------------
# MLX-LM: names a topic in a window of transcript
# --------------------------------------------------------------------------

# Offsets, not video time: a window is a few minutes of a two-hour file, and a
# small model handed absolute timestamps reliably invents them. Relative ones it
# can read straight off the [mm:ss] stamps.
#
# One plain line per chapter rather than JSON. mlx-lm has no constrained decoding
# (its logits processors are penalties only), so the reply shape has to come
# from the prompt -- and a line format is both easier for a small model to hold
# and safe to cut off: a JSON reply truncated by max_tokens throws away every
# chapter in it, a line reply keeps the ones it finished writing.
#
# The worked example is deliberately off-topic, and must stay that way. Measured
# against this transcript on Llama-3.2-3B, an example titled after the subject
# ("Why the scheduler reads ahead") got copied verbatim into the reply in 2 of 12
# runs; the bread-baking example, in 0 of 12.
_CHAPTER_PROMPT = """Split this transcript excerpt into video chapters.

Reply with 1-{max} lines and nothing else, each in exactly this form:
  <mm:ss> - <title>
where <mm:ss> is the stamp of the first transcript line of that chapter and
<title> is 3-8 words naming what is discussed from that point on. To show the
shape only, here is the reply to an unrelated talk about bread baking:
  00:00 - Why the starter needs feeding twice a day
  01:47 - Shaping and scoring the loaf
Those two lines are an example, not an answer: title THIS transcript instead.

The stamps are offsets from the start of this excerpt, so the first one is
00:00. No numbering, no quotes, no trailing punctuation, and never reuse the
previous chapter's title.

Cut where the topic actually changes, not at even intervals. If the excerpt is
one continuous topic, reply with a single line starting at 00:00.

Previous chapter: {prev}

Transcript:
{text}"""

# The stamp is unbracketed and the title follows a dash, so a line quoted back
# out of the transcript (those look like `[01:23] text`) cannot be mistaken for
# a chapter. Horizontal space only: `\s` would let a title-less line swallow the
# newline and merge itself with the next chapter.
_CHAPTER_LINE = re.compile(
    r"^[ \t]*(\d{1,3}):([0-5]\d)[ \t]+[-–—][ \t]+(.+?)[ \t]*$", re.MULTILINE)
_TITLE_EDGES = re.compile(r"^[\s\"'“‘]+|[\s\"'”’.,;:!?]+$")


class MLXLLM:
    """Chapter titles for a window of transcript, from a local MLX model.

    Weights load on the first window and then stay resident for the life of the
    process. There is no keep-alive to expire them the way a served model has,
    so the cost is unified memory held beside Whisper — see the README.
    """

    def __init__(self, model: str = CHAPTER_MODEL,
                 max_chapters: int = CHAPTER_MAX_PER_WINDOW,
                 max_tokens: int = 256):
        self.model = model
        self.max_chapters = max_chapters
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def has_model(self) -> bool:
        """Are the weights already on disk?

        Same reason the ollama probe existed: the first `chapters()` call on a
        repo that isn't cached pulls a couple of GB nobody asked for, so the
        chapterer stays off until they are there. Asking only about `*.safetensors`
        keeps a half-populated cache (tokenizer fetched, weights not) from reading
        as ready, and `local_files_only` makes it a cache stat — no network, and
        nothing to serialise against a generation in flight.
        """
        target = resolve_model(self.model)
        if Path(target).is_dir():
            return True
        try:
            snapshot_download(target, local_files_only=True,
                              allow_patterns=["*.safetensors"])
            return True
        except Exception:
            return False

    def chapters(self, text: str, prev: str | None = None) -> list[dict]:
        """[{start, title}] with `start` in seconds from the start of `text`."""
        model, tokenizer = self._load()
        template = {"tokenize": False, "add_generation_prompt": True}
        # A reasoning model would otherwise spend its whole budget thinking about
        # titles. apply_chat_template remaps or drops this for models with no
        # thinking channel.
        template["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template([{
            "role": "user",
            "content": _CHAPTER_PROMPT.format(
                max=self.max_chapters, prev=prev or "(none)", text=text),
        }], **template)

        # ponytail: no lock around the load. Chapterer runs a single worker
        # thread, so `chapters()` has one caller; add one if a second lands.
        reply = generate(model, tokenizer, prompt, max_tokens=self.max_tokens,
                         sampler=make_sampler(temp=0.4))

        out = []
        for minutes, seconds, title in _CHAPTER_LINE.findall(reply):
            title = _TITLE_EDGES.sub("", title)[:80]
            if title:
                out.append({"start": int(minutes) * 60 + int(seconds),
                            "title": title})
        return sorted(out, key=lambda c: c["start"])[:self.max_chapters]

    def _load(self):
        """Load the weights once, lazily.

        The first window pays a cold load (seconds from disk). That is fine here
        and nowhere else: Chapterer runs this on its own worker thread, so the
        load cannot stall the playhead the way a request thread would.
        """
        if self._model is None:
            self._model, self._tokenizer = load(resolve_model(self.model))
        return self._model, self._tokenizer


_llm: MLXLLM | None = None


def get_llm() -> MLXLLM:
    """The one resident LLM client. Cheap to build: it holds no weights yet."""
    global _llm
    if _llm is None:
        _llm = MLXLLM()
    return _llm
