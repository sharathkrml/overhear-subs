"""No-model tests for the read-along synthesiser: dedup, speed retry, caching."""

import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from backends import speech_text
from pipeline import Cue, TTSSynthesizer

SR = 24000


def synth(tmp: Path, calls: list, samples: int = SR // 10):
    """A synthesizer that records what it was asked for and always returns the
    same length of audio, so the cue's span alone decides the speed retry."""
    def speak(text, speed: float = 1.0):
        calls.append((text, round(speed, 3)))
        return np.full(samples, 0.1, dtype=np.float32)
    return speak


def check(label: str, cond: bool, extra: object = "") -> None:
    assert cond, f"{label}: {extra}"


def settled(tts: TTSSynthesizer, timeout: float = 3.0) -> None:
    """Wait for the worker to drain its queue."""
    deadline = time.time() + timeout
    while time.time() < deadline and tts._queue.unfinished_tasks:
        time.sleep(0.02)


def build(tmp: Path, **kw):
    kw.setdefault("enabled", True)  # submit() gates on this, not on the thread
    tts = TTSSynthesizer(kw.pop("speak", lambda t, speed=1.0: None), SR, tmp, "af_heart", **kw)
    tts.start()  # the worker always runs; `enabled` only gates what work arrives
    return tts


# ------------------------------------------------------------------ speech_text


def test_speech_text_folds_newlines_so_kokoro_keeps_one_prosody():
    assert speech_text("The bridge\nwas closed.") == "The bridge was closed."


def test_speech_text_drops_whisper_noise_and_collapses_space():
    assert speech_text("  [Music] Hello   ♪ there  ") == "Hello there"


def test_speech_text_is_empty_when_nothing_is_speakable():
    assert speech_text("   ") == ""
    assert speech_text("♪") == ""


# ------------------------------------------------------------------ dedup


def test_a_rebroadcast_cue_is_not_spoken_twice():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=synth(Path(tmp), calls))
        tts.submit([Cue(0.0, 3.0, "once")])
        settled(tts)
        tts.submit([Cue(0.0, 3.0, "once")])  # a rewind, or a duplicate chunk
        settled(tts)
        tts.stop()
    assert [c[0] for c in calls] == ["once"]


def test_disabled_synthesiser_queues_nothing():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=synth(Path(tmp), calls), enabled=False)
        tts.submit([Cue(0.0, 3.0, "nope")])
        settled(tts)
        tts.stop()
    assert calls == []


# ------------------------------------------------------------------ speed retry


def test_audio_that_fits_the_cue_is_not_resynthesized():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=synth(Path(tmp), calls, samples=SR // 10))
        tts.submit([Cue(0.0, 0.2, "fits")])  # 0.1s of audio in a 0.2s cue
        settled(tts)
        tts.stop()
    assert calls == [("fits", 1.0)]


def test_audio_overrunning_its_cue_is_re_spoken_faster():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=synth(Path(tmp), calls, samples=SR // 10))
        tts.submit([Cue(0.0, 0.05, "tight")])  # 0.1s of audio in a 0.05s cue
        settled(tts)
        tts.stop()
    assert calls == [("tight", 1.0), ("tight", 1.6)]  # clamped at TTS_MAX_SPEED


# ------------------------------------------------------------------ cache


def test_path_for_is_voice_scoped_and_only_after_a_write():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tts = build(root, speak=synth(root, calls))
        assert tts.path_for(1.0) is None
        tts.submit([Cue(1.0, 4.0, "hello")])
        settled(tts)
        assert tts.path_for(1.0) == root / "af_heart" / "1000.wav"
        assert tts.path_for(2.0) is None
        tts.stop()


def test_switching_voice_requotes_everything_under_the_new_name():
    calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tts = build(root, speak=synth(root, calls))
        tts.submit([Cue(0.0, 3.0, "hi")])
        settled(tts)
        assert tts.path_for(0.0).parent.name == "af_heart"

        tts.set_voice("am_michael")  # re-queues every known cue by itself
        settled(tts)
        assert tts.path_for(0.0).parent.name == "am_michael"
        tts.stop()
    assert [c[0] for c in calls] == ["hi", "hi"]  # once per voice


def test_state_reports_ready_only_once_something_is_on_disk():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tts = build(root, speak=synth(root, []))
        assert tts.state()["ready"] is False
        tts.submit([Cue(0.0, 3.0, "hello")])
        settled(tts)
        assert tts.state()["ready"] is True
        assert tts.state()["error"] is None
        tts.stop()


def test_spoken_counts_finished_wavs_not_merely_claimed_cues():
    """`spoken` drives the browser's "is there more audio yet" poll, so it must
    not tick up the moment a cue is queued — otherwise the client looks for
    files that aren't there and the cue is silently skipped."""
    release = threading.Event()

    def slow(text, speed: float = 1.0):
        release.wait(5.0)
        return np.full(SR // 10, 0.1, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=slow)
        tts.submit([Cue(0.0, 3.0, "held")])
        time.sleep(0.4)  # claimed and queued, not yet written
        held = tts.state()
        check("claimed but not written", held["spoken"] == 0 and held["ready"] is False, held)
        check("nothing on disk yet", tts.path_for(0.0) is None)
        release.set()
        settled(tts)
        check("written", tts.state()["spoken"] == 1, tts.state())
        tts.stop()


def test_a_failing_cue_reports_the_error_and_keeps_the_worker_alive():
    def boom(text, speed: float = 1.0):
        raise RuntimeError("no espeak")

    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=boom)
        tts.submit([Cue(0.0, 3.0, "one"), Cue(4.0, 7.0, "two")])
        settled(tts)
        state = tts.state()
        tts.stop()
    assert "no espeak" in state["error"]
    assert state["queued"] == 0  # both were attempted, not just the first


def test_a_cue_that_recovers_is_retried_not_lost():
    """The regression: `submit` claims a cue before queueing it and nothing ever
    re-submits one (its chunk is cached), so a single throw lost that line's audio
    for the rest of the session."""
    attempts = []

    def flaky(text, speed: float = 1.0):
        attempts.append(text)
        if len(attempts) == 1:
            raise RuntimeError("kokoro warming up")
        return np.full(SR // 10, 0.1, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=flaky)
        tts.submit([Cue(0.0, 3.0, "one")])
        settled(tts)
        state = tts.state()
        path = tts.path_for(0.0)
        written = path is not None and path.is_file()
        tts.stop()
    assert len(attempts) == 2, "the retry did not happen"
    assert state["spoken"] == 1, "the recovered cue was never written"
    assert written


def test_a_cue_that_always_throws_gives_up_instead_of_spinning():
    """Same ceiling as the scheduler: bounded, or one bad cue starves every later
    one because they share a single worker and a single queue."""
    attempts = []

    def boom(text, speed: float = 1.0):
        attempts.append(text)
        raise RuntimeError("no espeak")

    with tempfile.TemporaryDirectory() as tmp:
        tts = build(Path(tmp), speak=boom)
        tts.submit([Cue(0.0, 3.0, "one")])
        settled(tts)
        state = tts.state()
        tts.stop()
    assert len(attempts) == TTSSynthesizer._TRIES, "a doomed cue retried forever"
    assert state["queued"] == 0
    assert state["spoken"] == 0


# ------------------------------------------------------------------ wav


def test_written_wav_is_playable_16bit_mono():
    with tempfile.TemporaryDirectory() as tmp:
        calls: list = []
        root = Path(tmp)
        tts = build(root, speak=synth(root, calls, samples=SR // 2))
        tts.submit([Cue(0.0, 3.0, "hello")])
        settled(tts)
        tts.stop()
        with wave.open(str(tts.path_for(0.0)), "rb") as fh:
            assert (fh.getnchannels(), fh.getsampwidth(), fh.getframerate()) == (1, 2, SR)
            frames = fh.readframes(fh.getnframes())
    peak = max(abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
               for i in range(0, len(frames), 2))
    assert peak > 28000  # normalised to the target, not left at Kokoro's ~0.29
