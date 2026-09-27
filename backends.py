"""Whisper backend: auto-detect the spoken language, translate to English.

Whisper's built-in `task="translate"` does any-language -> English in one pass,
so there is a single backend and no separate translation stage.

Kokoro speaks those translated cues back for read-along. It needs espeak-ng as
its grapheme-to-phoneme step, like every small open English TTS.
"""

from __future__ import annotations

import os
import re
import threading
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
