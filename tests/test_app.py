"""Session-lifecycle tests for the FastAPI layer: the cancellation generation,
teardown, and export formatting. No ffmpeg, no models, no network."""

import pytest
from fastapi import HTTPException

import app as app_mod
from app import OpenReq, Session, _disposition, _on_cues, _text, _ts
from pipeline import Cue


# ------------------------------------------------------- cancellation generation


class _Recorder:
    def __init__(self):
        self.submitted = []

    def submit(self, cues):
        self.submitted.extend(cues)


def test_cues_from_a_cancelled_session_are_dropped(monkeypatch):
    """The regression: a whisper chunk outlives `stop()`'s 5s join, so the
    abandoned video's callback fires *after* the next session is installed.
    `_on_cues` read the global `session`, so those cues were submitted to the
    new session's TTS and pushed over the websocket as transcript rows belonging
    to a video the user already closed."""
    fake = Session()
    tts, chapters = _Recorder(), _Recorder()
    fake.tts, fake.chapters, fake.loop, fake.clients = tts, chapters, None, set()
    monkeypatch.setattr(app_mod, "session", fake)

    monkeypatch.setattr(app_mod, "_open_gen", 7)
    _on_cues(0, [Cue(0.0, 1.0, "a", "b")], gen=7)  # current generation: kept
    assert len(tts.submitted) == 1 and len(chapters.submitted) == 1

    _on_cues(1, [Cue(1.0, 2.0, "c", "d")], gen=6)  # stale: dropped
    assert len(tts.submitted) == 1, "a stale chunk reached the new session"
    assert len(chapters.submitted) == 1


def test_on_cues_without_a_generation_is_unconditional(monkeypatch):
    """The default keeps the old direct-call behaviour; only sessions built by
    /api/open pass a generation, and those are the ones that need fencing."""
    fake = Session()
    tts = _Recorder()
    fake.tts, fake.loop, fake.clients = tts, None, set()
    monkeypatch.setattr(app_mod, "session", fake)
    _on_cues(0, [Cue(0.0, 1.0, "a", "b")])
    assert len(tts.submitted) == 1


# ------------------------------------------------------------------ teardown


def test_teardown_stops_every_worker_and_clears_the_session(monkeypatch):
    stopped = []

    class Worker:
        def __init__(self, name):
            self.name = name

        def stop(self):
            stopped.append(self.name)

    class Playback(Worker):
        def cancel(self):
            stopped.append(self.name + ":cancel")

    fake = Session()
    fake.path = __import__("pathlib").Path("/tmp/clip.mkv")
    fake.playback, fake.tts, fake.chapters, fake.scheduler = (
        Playback("playback"), Worker("tts"), Worker("chapters"), Worker("scheduler"))
    fake.prep = object()
    monkeypatch.setattr(app_mod, "session", fake)

    app_mod._teardown()

    assert "playback:cancel" in stopped, "an in-flight ffmpeg remux survives teardown"
    for name in ("tts", "chapters", "scheduler"):
        assert name in stopped, f"{name} kept running after teardown"
    assert fake.path is None and fake.scheduler is None
    assert fake.playback is None and fake.prep is None


# --------------------------------------------------------------- open locking


def _clip(tmp_path):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"not really a video")
    return path


def test_a_second_open_is_refused_while_one_is_running(tmp_path):
    """The regression: two concurrent opens both ran ffmpeg into the *same* temp
    names -- the PCM cache is keyed by file identity and the remux writes
    <key>.part.mp4 -- so they clobbered each other. The generation guard kept the
    loser from installing its session but not from corrupting the cache."""
    clip = _clip(tmp_path)
    assert app_mod._OPEN_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(HTTPException) as caught:
            app_mod.open_media(OpenReq(path=str(clip)))
        assert caught.value.status_code == 409
    finally:
        app_mod._OPEN_LOCK.release()


def test_a_failed_open_releases_the_lock(tmp_path, monkeypatch):
    """Otherwise the first bad video wedges the app: every later open 409s."""
    clip = _clip(tmp_path)

    def boom(path):
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(app_mod, "_open", boom)
    with pytest.raises(RuntimeError):
        app_mod.open_media(OpenReq(path=str(clip)))
    assert not app_mod._OPEN_LOCK.locked()


def test_a_missing_file_is_rejected_before_anything_else(tmp_path):
    with pytest.raises(HTTPException) as caught:
        app_mod.open_media(OpenReq(path=str(tmp_path / "nope.mkv")))
    assert caught.value.status_code == 404
    assert not app_mod._OPEN_LOCK.locked()


# -------------------------------------------------------------------- export


def test_timestamps_use_srt_commas_and_vtt_periods():
    assert _ts(0.0) == "00:00:00,000"
    assert _ts(3661.5, ".") == "01:01:01,500".replace(",", ".")
    # A cue that starts at t=0 must not go negative, and rounding must not
    # overflow into the next second.
    assert _ts(-5.0) == "00:00:00,000"
    assert _ts(59.9999) == "00:01:00,000"


def test_export_text_prefers_the_translation_as_a_second_line():
    translated = Cue(0.0, 1.0, "hello", "bonjour")
    assert _text(translated) == "hello\nbonjour"
    # Nothing to distinguish them: one line, not a duplicated blank one.
    same = Cue(0.0, 1.0, "hello", "hello")
    assert _text(same) == "hello"
    # Whisper occasionally emits a target with no source.
    assert _text(Cue(0.0, 1.0, "", "bonjour")) == "bonjour"


def test_export_is_served_as_a_download():
    assert _disposition("transcript.srt") == {
        "Content-Disposition": 'attachment; filename="transcript.srt"'}