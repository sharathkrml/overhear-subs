"""Media prep, chunk planning, and the look-ahead scheduler.

The whole trick: a local video's audio track is random-access once demuxed to
raw PCM. Transcription is just an index into a memmap, so we can transcribe
whatever is about to play before the playhead gets there. No streaming ASR, no
ring buffer, no VAD.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CACHE_DIR = Path(os.environ.get("LT_CACHE", Path.home() / ".cache" / "overhear-subs"))

# Shorter chunks mean a seek waits less for the in-flight chunk to clear;
# whisper's native window is 30s. Tunable via LT_CHUNK.
CHUNK_TARGET = float(os.environ.get("LT_CHUNK", "30.0"))
CHUNK_MAX = CHUNK_TARGET
SILENCE_SNAP = 6.0  # how far a boundary may drift to land on a silence
LOOKAHEAD = float(os.environ.get("LT_LOOKAHEAD", "10.0"))

VIDEO_EXT = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi",
    ".ts", ".flv", ".wmv", ".mpg", ".mpeg",
}
# Containers the browser can play as-is, given browser-safe codecs inside.
CONTAINER_OK = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {"aac", "mp3", "opus", "vorbis"}


@dataclass
class Cue:
    """One subtitle line. `target` is empty when no translation ran."""

    start: float
    end: float
    source: str
    target: str = ""


# --------------------------------------------------------------------------
# ffmpeg media prep
# --------------------------------------------------------------------------


def _cache_key(video: Path) -> str:
    st = video.stat()
    raw = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _run_ffmpeg(args: list[str], duration: float = 0.0,
                on_progress: Callable[[float], None] | None = None) -> str:
    """Run ffmpeg, reporting fraction done from its own -progress lines.

    `-progress pipe:2` interleaves progress with the log on stderr, so one
    stream carries both (silencedetect output and progress arrive together).
    """
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    log: list[str] = []
    for line in proc.stderr:
        log.append(line)
        if on_progress and duration > 0 and line.startswith("out_time_ms="):
            try:
                micros = int(line.split("=", 1)[1] or 0)
            except ValueError:
                continue
            on_progress(min(1.0, micros / 1_000_000 / duration))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("".join(log[-5:]).strip() or "ffmpeg failed")
    return "".join(log)


def extract_pcm(video: Path, duration: float = 0.0,
                on_progress: Callable[[float], None] | None = None) -> Path:
    """Demux the audio track to mono 16k float32, cached by file identity."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{_cache_key(video)}.f32"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists() and out.stat().st_size > 0:
        return out
    _run_ffmpeg(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le",
         "-progress", "pipe:2", "-nostats", str(out)],
        duration, on_progress,
    )
    meta.write_text(json.dumps({"source": str(video)}))
    return out


def probe_codecs(video: Path) -> dict[str, str]:
    """First video/audio codec name per stream type, e.g. {"video": "hevc"}."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    codecs: dict[str, str] = {}
    for stream in json.loads(proc.stdout).get("streams", []):
        codecs.setdefault(stream.get("codec_type", ""), stream.get("codec_name", ""))
    return codecs


def _needs_conversion(video: Path, codecs: dict[str, str]) -> str | None:
    """Why this file can't be played as-is, or None if it can."""
    if video.suffix.lower() not in CONTAINER_OK:
        return "container"
    video_codec = codecs.get("video")
    if video_codec not in BROWSER_VIDEO:
        return f"video codec '{video_codec or 'none'}'"
    audio_codec = codecs.get("audio")
    if audio_codec is not None and audio_codec not in BROWSER_AUDIO:
        return f"audio codec '{audio_codec}'"
    return None


@lru_cache(maxsize=1)
def _video_encoder() -> str:
    """Hardware H.264 when the ffmpeg build has it, else libx264."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                          capture_output=True, text=True)
    return "h264_videotoolbox" if "h264_videotoolbox" in proc.stdout else "libx264"


def _duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


class PlaybackPrep:
    """Produces a browser-playable mp4, in the background if it must transcode.

    Codec matters as much as container: an HEVC .mp4 looks playable by
    extension but Chromium refuses the streams, so the player just dies. We
    probe first and only re-encode what the browser can't already decode.
    """

    def __init__(self, source: Path):
        self.source = source
        self.output: Path | None = None
        self.error: str | None = None
        self.reason: str | None = None
        self.progress = 0.0
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._proc: subprocess.Popen | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if os.environ.get("LT_REMUX", "1") == "0":
            return self._finish(self.source)
        try:
            codecs = probe_codecs(self.source)
            self.reason = _needs_conversion(self.source, codecs)
        except Exception as exc:
            self.error = f"could not probe streams: {exc}"
            self._ready.set()
            return

        if self.reason is None:
            return self._finish(self.source)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"{_cache_key(self.source)}.mp4"
        if cache.exists() and cache.stat().st_size > 0 and self._cache_is_good(cache):
            return self._finish(cache)
        cache.unlink(missing_ok=True)

        self._thread = threading.Thread(target=self._run, args=(codecs, cache), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Abandon an in-flight conversion; safe to call at any point."""
        self._cancelled.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.kill()

    def _finish(self, output: Path) -> None:
        self.output = output
        self._ready.set()

    def _cache_is_good(self, cache: Path) -> bool:
        """Never trust a cache written by an older ffmpeg invocation."""
        try:
            return _needs_conversion(cache, probe_codecs(cache)) is None
        except Exception:
            return False

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def state(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "reason": self.reason,
            "progress": round(self.progress, 3),
            "converted": bool(self.output) and self.output != self.source,
        }

    # -- conversion --------------------------------------------------------

    def _run(self, codecs: dict[str, str], cache: Path) -> None:
        part = cache.with_name(cache.stem + ".part.mp4")
        try:
            duration = _duration(self.source)
            proc = subprocess.Popen(
                _convert_args(self.source, part, codecs),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._proc = proc
            for line in proc.stdout:
                if self._cancelled.is_set():
                    proc.kill()
                    break
                if line.startswith("out_time_ms=") and duration > 0:
                    micros = int(line.split("=", 1)[1] or 0)
                    self.progress = min(1.0, micros / 1_000_000 / duration)
            proc.wait()
            if self._cancelled.is_set():
                part.unlink(missing_ok=True)
                return
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr.read() or "ffmpeg failed").strip()[-300:])
            part.replace(cache)
            self.progress = 1.0
            self.output = cache
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            part.unlink(missing_ok=True)
        finally:
            self._proc = None
            self._ready.set()


def _convert_args(video: Path, out: Path, codecs: dict[str, str]) -> list[str]:
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if codecs.get("video") in BROWSER_VIDEO:
        args += ["-c:v", "copy"]
    elif _video_encoder() == "h264_videotoolbox":
        # ponytail: bitrate-capped hardware encode, no quality tuning.
        # Swap to libx264 -crf if a file comes out visibly soft.
        # -g 48: a short GOP so scrubbing only decodes ~2s back to a keyframe.
        args += ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-g", "48"]
    else:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                 "-g", "48", "-keyint_min", "48", "-sc_threshold", "0"]
    audio = codecs.get("audio")
    if audio is None:
        pass
    elif audio in BROWSER_AUDIO:
        args += ["-c:a", "copy"]
    else:
        args += ["-c:a", "aac", "-b:a", "192k"]
    return args + ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(out)]


_SILENCE = re.compile(r"silence_(start|end):\s*(-?[\d.]+)")


def detect_silences(video: Path, duration: float = 0.0,
                    on_progress: Callable[[float], None] | None = None) -> list[float]:
    """Midpoints of silence regions, used to avoid cutting mid-word."""
    stderr = _run_ffmpeg(
        ["ffmpeg", "-v", "info", "-i", str(video),
         "-af", "silencedetect=n=-35dB:d=0.4", "-f", "null",
         "-progress", "pipe:2", "-nostats", "-"],
        duration, on_progress,
    )
    starts: list[float] = []
    mids: list[float] = []
    for kind, value in _SILENCE.findall(stderr):
        t = float(value)
        if t < 0:
            continue
        if kind == "start":
            starts.append(t)
        elif starts:
            mids.append((starts.pop(0) + t) / 2.0)
    return mids


def plan_chunks(
    duration: float,
    silences: list[float],
    target: float = CHUNK_TARGET,
    max_len: float = CHUNK_MAX,
    snap: float = SILENCE_SNAP,
) -> list[tuple[float, float]]:
    """Contiguous [t0, t1) chunks of at most `max_len`, edges snapped to silence."""
    points = sorted(silences)
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.05:
        if duration - start <= max_len:
            chunks.append((start, duration))
            break
        target_end = start + target
        end = _snap(target_end, points, snap)
        if not (start + 1.0 < end <= start + max_len):
            end = min(target_end, duration)
        chunks.append((start, end))
        start = end
    return chunks


def _snap(t: float, points: list[float], tol: float) -> float:
    best, best_d = t, tol
    for p in points:
        d = abs(p - t)
        if d <= best_d:
            best, best_d = p, d
    return best


# --------------------------------------------------------------------------
# subtitle reflow: one cue = max 2 lines on video
# --------------------------------------------------------------------------

SUB_LINE_CHARS = 42  # per line, incl. CJK chars
# ponytail: 80 not 84 — halves balance at ~40, so a word on the seam can't
# push either line past ~42 into a wrapped 3rd line.
SUB_MAX_CHARS = 80


def _greedy(text: str) -> list[str]:
    """Fewest pieces of ≤ SUB_MAX_CHARS, preferring word boundaries."""
    text = " ".join(text.split())
    if not text:
        return []
    if " " not in text:  # CJK / one long word: hard cut
        return [text[i:i + SUB_MAX_CHARS] for i in range(0, len(text), SUB_MAX_CHARS)]
    pieces, cur_len, cur = [], 0, []
    for w in text.split(" "):
        while len(w) > SUB_MAX_CHARS:  # pathological word: hard cut it
            if cur:
                pieces.append(" ".join(cur))
                cur, cur_len = [], 0
            pieces.append(w[:SUB_MAX_CHARS])
            w = w[SUB_MAX_CHARS:]
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > SUB_MAX_CHARS:
            pieces.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _slice_like(text: str, sizes: list[int]) -> list[str]:
    """Cut `text` into len(sizes) pieces ∝ sizes, snapping to word edges."""
    text = " ".join(text.split())
    total = sum(sizes)
    if not text or not total:
        return [""] * len(sizes)
    bounds = []
    pos = 0.0
    for s in sizes[:-1]:
        pos += s
        bounds.append(int(round(pos / total * len(text))))
    pieces, prev = [], 0
    for b in bounds:
        b = max(prev + 1, min(len(text) - 1, b)) if prev + 1 < len(text) else len(text)
        snap = text.rfind(" ", prev + 1, b + 11)  # nearest space, slight lookahead
        cut = snap if snap > prev else b
        pieces.append(text[prev:cut].strip())
        prev = cut
    pieces.append(text[prev:].strip())
    return pieces


def _balance(text: str) -> str:
    """One `\n` near the middle so the overlay renders as 2 balanced lines."""
    if len(text) <= SUB_LINE_CHARS or "\n" in text:
        return text
    mid = len(text) // 2
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if spaces:
        i = min(spaces, key=lambda s: (abs(s - mid), s))
        return text[:i] + "\n" + text[i + 1:]
    return text[:mid] + "\n" + text[mid:]


def reflow_cues(cues: list[Cue]) -> list[Cue]:
    """Split long cues so every cue fits 1-2 lines; time split ∝ text length."""
    out: list[Cue] = []
    for cue in cues:
        src = _greedy(cue.source)
        tgt = _greedy(cue.target) if cue.target else []
        if len(src) <= 1 and len(tgt) <= 1:
            cue.source = _balance(cue.source)
            if cue.target:
                cue.target = _balance(cue.target)
            out.append(cue)
            continue
        # Boundaries follow the longer side; the shorter side is sliced ∝.
        if len(tgt) > len(src):
            sizes = [len(p) for p in tgt]
            src = _slice_like(cue.source, sizes)
        else:
            sizes = [len(p) for p in src]
            tgt = _slice_like(cue.target, sizes) if cue.target else [""] * len(src)
        weights = [max(1, len(s) + len(t)) for s, t in zip(src, tgt)]
        span, t = cue.end - cue.start, cue.start
        total = sum(weights)
        for i, (s, tg, w) in enumerate(zip(src, tgt, weights)):
            end = cue.end if i == len(src) - 1 else t + span * w / total
            out.append(Cue(t, end, _balance(s), _balance(tg)))
            t = end
    return out


class AudioSource:
    """Random-access float32 mono audio."""

    def __init__(self, pcm_path: Path):
        self.data = np.memmap(pcm_path, dtype="<f4", mode="r")

    @property
    def duration(self) -> float:
        return len(self.data) / SAMPLE_RATE

    def slice(self, t0: float, t1: float) -> np.ndarray:
        a = max(0, int(t0 * SAMPLE_RATE))
        b = min(len(self.data), int(t1 * SAMPLE_RATE))
        return np.array(self.data[a:b], dtype=np.float32)


def build_media(
    video: Path,
    on_progress: Callable[[str, float], None] | None = None,
) -> tuple[AudioSource, float, list[tuple[float, float]]]:
    """Extract audio + plan chunks, reporting ("audio"|"silence", fraction)."""
    report = on_progress or (lambda stage, frac: None)
    duration = _duration(video)
    pcm = extract_pcm(video, duration, lambda f: report("audio", f))
    report("audio", 1.0)  # a cache hit jumps straight here
    source = AudioSource(pcm)
    duration = source.duration
    silences = detect_silences(video, duration, lambda f: report("silence", f))
    return source, duration, plan_chunks(duration, silences)


# --------------------------------------------------------------------------
# look-ahead scheduler
# --------------------------------------------------------------------------


class LookaheadScheduler:
    """Transcribes forward from the playhead, then back-fills behind it.

    A seek moves `next_idx` to the target, so transcription resumes there and
    runs to the end before sweeping up chunks skipped over (e.g. seek to 10:00:
    do 10:00→finish, then 0:00→10:00). Results are cached by chunk index, so a
    backward seek re-serves them instantly.
    """

    def __init__(
        self,
        chunks: list[tuple[float, float]],
        run_chunk,
        lookahead: float = LOOKAHEAD,
        on_cues=None,
        poll: float = 0.2,
    ):
        self.chunks = list(chunks)
        self.run_chunk = run_chunk
        self.lookahead = lookahead
        self.on_cues = on_cues or (lambda idx, cues: None)
        self.poll = poll

        self.cache: dict[int, list[Cue]] = {}
        self.next_idx = 0
        self.playhead = 0.0
        self.error: str | None = None
        self.warming = bool(self.chunks)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def set_playhead(self, t: float) -> None:
        with self._lock:
            self.playhead = t
            idx = self._index_at(t)
            if idx is not None:
                self.next_idx = idx

    def state(self) -> dict:
        with self._lock:
            until = self._transcribed_until()
            done = len(self.cache)
            return {
                "playhead": round(self.playhead, 2),
                "transcribed_until": round(until, 2),
                "ahead": round(until - self.playhead, 2),
                "lookahead": self.lookahead,
                "chunks_done": done,
                "chunks_total": len(self.chunks),
                "cached": sorted(self.cache),
                "finished": done >= len(self.chunks),
                "error": self.error,
                "warming": self.warming and done == 0 and self.error is None,
            }

    def all_cues(self) -> list[Cue]:
        with self._lock:
            out: list[Cue] = []
            for idx in sorted(self.cache):
                out.extend(self.cache[idx])
        return sorted(out, key=lambda c: c.start)

    # -- internals ---------------------------------------------------------

    def _index_at(self, t: float) -> int | None:
        for i, (a, b) in enumerate(self.chunks):
            if a <= t < b:
                return i
        return None

    def _transcribed_until(self) -> float:
        """End of the contiguous transcribed run containing the playhead."""
        idx = self._index_at(self.playhead)
        if idx is None:
            return self.chunks[-1][1] if self.chunks else 0.0
        j = idx
        while j in self.cache:
            j += 1
        return self.chunks[j - 1][1] if j > idx else self.chunks[idx][0]

    def _pick(self) -> int | None:
        """Next chunk to run: forward from the playhead, then anything behind.

        Forward first so a seek's target streams immediately; only once there's
        nothing left ahead do we sweep up chunks skipped by earlier seeks.
        """
        with self._lock:
            idx = self.next_idx
            while idx < len(self.chunks) and idx in self.cache:
                idx += 1
            self.next_idx = idx
            if idx < len(self.chunks):
                return idx
            for i, _ in enumerate(self.chunks):
                if i not in self.cache:
                    return i
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            idx = self._pick()
            if idx is None:
                self._stop.wait(self.poll)
                continue

            t0, t1 = self.chunks[idx]
            try:
                cues = reflow_cues(self.run_chunk(idx, t0, t1))
            except Exception as exc:  # keep the worker alive; surface to the UI
                self.warming = False
                self.error = f"chunk {idx}: {type(exc).__name__}: {exc}"
                self._stop.wait(1.0)
                continue

            self.warming = False
            self.error = None
            with self._lock:
                self.cache[idx] = cues
                if self.next_idx == idx:
                    self.next_idx = idx + 1
            try:
                self.on_cues(idx, cues)
            except Exception:
                pass


# --------------------------------------------------------------------------
# read-along TTS
# --------------------------------------------------------------------------

TTS_PEAK = 0.95  # normalise so every cue lands at a consistent loudness
TTS_MAX_SPEED = 1.6


def write_wav(path: Path, audio, sample_rate: int) -> None:
    """Mono float32 -> 16-bit PCM wav, written atomically.

    The browser polls for these files, so a half-written one would decode as
    noise. Write beside the target and rename.
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if 0.01 < peak < 1.0:  # leave silence alone, don't amplify hiss
        pcm = (np.clip(audio / peak * TTS_PEAK, -1.0, 1.0) * 32767.0).astype("<i2")

    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    os.replace(tmp, path)


class TTSSynthesizer:
    """Speaks cues ahead of the playhead into the same cache an export would read.

    Kokoro runs ~12x realtime, so the ten seconds of runway the ASR scheduler
    already keeps is plenty: a cue is on disk long before the playhead reaches
    it. Cues are keyed by start time, so a re-broadcast (a rewind, a duplicate
    chunk) is free and a resumed session reuses the cache.
    """

    def __init__(self, speak, sample_rate: int, root: Path, voice: str,
                 enabled: bool = False):
        self.speak = speak
        self.sample_rate = sample_rate
        self.root = Path(root)
        self.voice = voice
        self.enabled = enabled

        self.cues: dict[int, Cue] = {}
        self.claimed: set[int] = set()  # queued or written: dedup set
        self.written = 0                # wavs actually on disk
        self.error: str | None = None
        self._queue: queue.Queue[Cue | None] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="tts")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        # Always start: `enabled` gates what work arrives, not the thread
        # itself, so toggling read-along on later needs no restart dance.
        self._thread.start()

    def set_voice(self, voice: str) -> None:
        """Switch voice mid-session. Each voice caches under its own directory,
        so switching back replays instantly instead of re-synthesising."""
        with self._lock:
            if voice == self.voice:
                return
            self.voice = voice
            self.claimed.clear()
            # `written` is scoped to the cache directory, and that's what the
            # browser fetches from, so it restarts with the voice.
            self.written = 0
            pending = list(self.cues.values())
        self.submit(pending)  # re-speak everything under the new voice

    def stop(self) -> None:
        self._stop.set()
        for _ in range(64):  # bounded: a drain in flight finishes on its own
            if self._queue.empty():
                break
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # -- work --------------------------------------------------------------

    def submit(self, cues: list[Cue]) -> None:
        """Queue whatever hasn't been claimed yet. Cues with nothing speakable
        are claimed too, so they aren't reconsidered on every re-broadcast."""
        if not self.enabled or self._stop.is_set():
            return
        for cue in cues:
            key = self._key(cue)
            with self._lock:
                if key in self.claimed:
                    continue
                self.claimed.add(key)
                self.cues[key] = cue
            self._queue.put(cue)

    def _dir(self) -> Path:
        return self.root / self.voice

    def path_for(self, start: float) -> Path | None:
        path = self._dir() / f"{self._key(start)}.wav"
        return path if path.is_file() else None

    def state(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "ready": self.written > 0,
                "voice": self.voice,
                # Counts finished wavs, not claimed cues: the browser uses this
                # to decide when to look for more audio.
                "spoken": self.written,
                "queued": self._queue.qsize(),
                "error": self.error,
            }

    # -- internals ---------------------------------------------------------

    def _key(self, cue) -> int:
        return int(round((cue.start if hasattr(cue, "start") else cue) * 1000))

    def _one(self, cue: Cue) -> None:
        text = cue.target or cue.source
        audio = self.speak(text)
        if audio is None:
            return
        # A cue shorter than the line it holds would cut the audio off mid-word.
        # One retry at a higher speed, then accept the overlap: at 10s of runway
        # the second pass is free, and chopping audio is worse than a little
        # bleed into the next cue.
        span = max(0.05, cue.end - cue.start)
        if audio.size / self.sample_rate > span:
            faster = self.speak(text, speed=min(TTS_MAX_SPEED, audio.size / self.sample_rate / span))
            if faster is not None:
                audio = faster
        out = self._dir()
        out.mkdir(parents=True, exist_ok=True)
        write_wav(out / f"{self._key(cue)}.wav", audio, self.sample_rate)
        with self._lock:
            self.written += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                cue = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if cue is not None:
                    self._one(cue)
            except Exception as exc:  # a bad cue must not kill the worker
                with self._lock:
                    self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()
