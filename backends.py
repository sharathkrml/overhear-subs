"""Whisper backend: auto-detect the spoken language, translate to English.

Whisper's built-in `task="translate"` does any-language -> English in one pass,
so there is a single backend and no separate translation stage.

Kokoro speaks those translated cues back for read-along. It needs espeak-ng as
its grapheme-to-phoneme step, like every small open English TTS.

Ollama names a topic every so often so the panel gets a chapter list. It talks
HTTP rather than loading weights, so there is nothing to warm: the only
question is whether the model is already pulled.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from pathlib import Path

import numpy as np

from pipeline import SAMPLE_RATE, Cue

# large-v3-turbo silently ignores task="translate" (it just transcribes), so use
# the full large-v3 weights, which are translate-capable.
TRANSLATE_ASR = os.environ.get("LT_TRANSLATE_MODEL", "mlx-community/whisper-large-v3-mlx")

# Kokoro is the fastest of the good small English voices and measured 12.7x
# realtime on an M1 Pro, so a cue is spoken in ~4% of the look-ahead window.
# 4bit is indistinguishable from bf16 in speed and ~20% lighter.
TTS_MODEL = os.environ.get("LT_TTS_MODEL", "mlx-community/Kokoro-82M-4bit")
TTS_VOICE = os.environ.get("LT_TTS_VOICE", "af_heart")
TTS_LANG = os.environ.get("LT_TTS_LANG", "a")  # 'a' = American English

# Ollama serves the chapter titles. Anything local works; a small instruct model
# is enough, because the only job is naming a topic from ~1.5k words of context.
OLLAMA_URL = os.environ.get("LT_OLLAMA_URL", "http://127.0.0.1:11434")
CHAPTER_MODEL = os.environ.get("LT_CHAPTER_MODEL", "qwen3:8b")
CHAPTER_MAX_PER_WINDOW = 4
# A cold qwen3:8b load is ~30s, so the request has to outlast it. The 10s of
# transcript runway the ASR scheduler already keeps is unaffected: the worker
# thread is not the one feeding the playhead.
CHAPTER_TIMEOUT = 300.0


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


_backend: WhisperASR | None = None


def get_backend() -> WhisperASR:
    """The one resident model: auto-detect + translate to English."""
    global _backend
    if _backend is None:
        _backend = WhisperASR(TRANSLATE_ASR, None, "translate")
    return _backend


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
# Ollama: names a topic in a window of transcript
# --------------------------------------------------------------------------

# Constrained decoding, so the reply is always this shape and nothing downstream
# has to parse prose or fish for a title in the middle of a sentence.
CHAPTER_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "title": {"type": "string"},
                },
                "required": ["start", "title"],
            },
        }
    },
    "required": ["chapters"],
}

_CHAPTER_SYSTEM = (
    "You split a transcript into chapters for a video's chapter list. "
    "Reply with JSON only."
)

# Offsets, not video time: a window is a few minutes of a two-hour file, and an
# 8B model handed absolute timestamps reliably invents them. Relative ones it can
# read straight off the [mm:ss] stamps.
_CHAPTER_PROMPT = """Split this transcript excerpt into video chapters.

Return 1-{max} chapters, each with:
- start: the [mm:ss] stamp of the first line of that chapter, in seconds. The
  stamps are offsets from the start of this excerpt, so the first line is 0.
- title: 3-8 words naming what is discussed from that point on. No quotes, no
  trailing punctuation, no numbering, and never reuse the previous chapter.

Cut where the topic actually changes, not at even intervals. If the excerpt is
one continuous topic, return a single chapter starting at 0.

Previous chapter: {prev}

Transcript:
{text}"""

# Models ignore the instruction often enough to be worth a cheap clean-up.
_TITLE_EDGES = re.compile(r"^[\s\"'“‘]+|[\s\"'”’.,;:!?]+$")


class OllamaLLM:
    """Chapter titles for a window of transcript, from a local Ollama server."""

    def __init__(self, url: str = OLLAMA_URL, model: str = CHAPTER_MODEL,
                 timeout: float = CHAPTER_TIMEOUT,
                 max_chapters: int = CHAPTER_MAX_PER_WINDOW):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_chapters = max_chapters
        self._lock = threading.Lock()

    def has_model(self) -> bool:
        """Is the model already pulled?

        /api/chat will quietly pull a missing model, and that is a multi-GB
        download nobody asked for, so the chapterer stays off until it is there.
        """
        try:
            with self._lock:
                tags = self._get("/api/tags", timeout=3.0)
        except Exception:
            return False
        names = {m.get("name") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def chapters(self, text: str, prev: str | None = None) -> list[dict]:
        """[{start, title}] with `start` in seconds from the start of `text`."""
        body = json.dumps({
            "model": self.model,
            "stream": False,
            # qwen3 is a reasoning model and will spend its budget thinking about
            # chapter titles instead of writing them.
            "think": False,
            # Evict after ten idle minutes instead of sitting in unified memory
            # for the rest of the session next to Whisper.
            "keep_alive": "10m",
            "format": CHAPTER_SCHEMA,
            "options": {"temperature": 0.4},
            "messages": [
                {"role": "system", "content": _CHAPTER_SYSTEM},
                {"role": "user", "content": _CHAPTER_PROMPT.format(
                    max=self.max_chapters, prev=prev or "(none)", text=text)},
            ],
        }).encode()
        with self._lock:
            reply = self._post("/api/chat", body)

        out = []
        for item in json.loads(reply["message"]["content"]).get("chapters") or []:
            start = item.get("start")
            title = _TITLE_EDGES.sub("", str(item.get("title") or ""))[:80]
            if isinstance(start, (int, float)) and not isinstance(start, bool) and title:
                out.append({"start": max(0.0, float(start)), "title": title})
        return sorted(out, key=lambda c: c["start"])[:self.max_chapters]

    # -- transport ---------------------------------------------------------

    def _get(self, path: str, timeout: float) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as fh:
            return json.loads(fh.read())

    def _post(self, path: str, body: bytes) -> dict:
        request = urllib.request.Request(
            self.url + path, data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as fh:
            return json.loads(fh.read())


_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    """The one resident LLM client. Cheap to build: it holds no weights."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm
