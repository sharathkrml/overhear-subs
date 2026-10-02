import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import (
    Cue,
    LookaheadScheduler,
    PlaybackPrep,
    _convert_args,
    _ffmpeg_budget,
    cache_entries,
    _needs_conversion,
    _run_ffmpeg,
    frame_jpeg,
    plan_chunks,
    prune_cache,
    remove_cache_key,
    reflow_cues,
)


# ------------------------------------------------------- playback formats


def test_playable_files_are_left_alone():
    mp4 = Path("clip.mp4")
    assert _needs_conversion(mp4, {"video": "h264", "audio": "aac"}) is None
    assert _needs_conversion(mp4, {"video": "vp9", "audio": "opus"}) is None
    assert _needs_conversion(mp4, {"video": "h264"}) is None  # no audio track


def test_unplayable_codecs_are_detected():
    mp4 = Path("clip.mp4")
    # The regression: an HEVC .mp4 looks fine by extension but Chromium
    # refuses the streams, so the player dies with a demuxer error.
    assert _needs_conversion(mp4, {"video": "hevc", "audio": "aac"}) == "video codec 'hevc'"
    assert _needs_conversion(mp4, {"video": "h264", "audio": "ac3"}) == "audio codec 'ac3'"
    assert _needs_conversion(mp4, {"video": "h264", "audio": "eac3"}) == "audio codec 'eac3'"
    assert _needs_conversion(mp4, {}) == "video codec 'none'"


def test_unplayable_containers_are_detected():
    assert _needs_conversion(Path("clip.mkv"), {"video": "h264", "audio": "aac"}) == "container"
    assert _needs_conversion(Path("clip.avi"), {"video": "h264", "audio": "mp3"}) == "container"


def test_reencode_sets_a_short_keyframe_interval(monkeypatch):
    out = Path("out.mp4")
    monkeypatch.setattr("pipeline._video_encoder", lambda: "libx264")
    reencoded = _convert_args(Path("clip.mkv"), out, {"video": "hevc", "audio": "aac"})
    assert "-g" in reencoded, "a long GOP makes every seek stall on decode"
    copied = _convert_args(Path("clip.mp4"), out, {"video": "h264", "audio": "aac"})
    assert "-g" not in copied, "copy must not touch the source GOP"


# -------------------------------------------------------------- cancelling


class _FakeProc:
    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True


def test_cancel_kills_in_flight_conversion():
    prep = PlaybackPrep(Path("clip.mkv"))
    proc = _FakeProc()
    prep._proc = proc
    prep.cancel()
    assert proc.killed is True
    assert prep._cancelled.is_set()


def test_cancel_is_safe_without_a_conversion():
    prep = PlaybackPrep(Path("clip.mkv"))
    prep.cancel()
    prep.cancel()
    assert prep._cancelled.is_set()
    assert prep.error is None


def test_remux_drains_stderr_into_the_stream_it_reads(monkeypatch, tmp_path):
    """The regression: stderr used to be a second pipe nobody drained.

    The loop reads stdout while ffmpeg fills the stderr pipe buffer, so a
    pathological file wedged the remux thread forever — and `cancel()`'s kill()
    raced a parent still blocked inside `for line in proc.stdout`.
    """
    seen = {}

    def fake_popen(args, **kw):
        seen.update(kw)
        Path(args[-1]).write_bytes(b"fake remux")

        class FakeProc:
            returncode = 0

            def __init__(self):
                self.stdout = iter(["out_time_ms=1000000\n", "not a progress line\n"])

            def wait(self):
                return 0

        return FakeProc()

    monkeypatch.setattr("pipeline.subprocess.Popen", fake_popen)
    monkeypatch.setattr("pipeline._duration", lambda p: 4.0)
    cache = tmp_path / "cache.mp4"
    prep = PlaybackPrep(Path("clip.mkv"))
    prep._run({"video": "h264", "audio": "aac"}, cache)
    assert seen.get("stderr") is subprocess.STDOUT
    assert prep.error is None, prep.error
    assert prep.progress == 1.0
    assert prep.output == cache and cache.is_file()


def test_remux_tolerates_an_unseekable_progress_stamp(monkeypatch, tmp_path):
    """ffmpeg prints `out_time_ms=N/A` when it cannot seek; that used to abort
    the whole conversion via the outer except, not just skip the line."""

    def fake_popen(args, **kw):
        Path(args[-1]).write_bytes(b"fake remux")

        class FakeProc:
            returncode = 0

            def __init__(self):
                self.stdout = iter(["out_time_ms=N/A\n", "out_time_ms=2000000\n"])

            def wait(self):
                return 0

        return FakeProc()

    monkeypatch.setattr("pipeline.subprocess.Popen", fake_popen)
    monkeypatch.setattr("pipeline._duration", lambda p: 4.0)
    prep = PlaybackPrep(Path("clip.mkv"))
    prep._run({"video": "h264", "audio": "aac"}, tmp_path / "cache.mp4")
    # Before the fix the N/A line raised ValueError into the outer except, which
    # recorded the error and deleted the partial output. Reaching 1.0 means the
    # whole loop ran and the remux landed.
    assert prep.error is None, prep.error
    assert prep.progress == 1.0
    assert (tmp_path / "cache.mp4").is_file()


# ------------------------------------------------------------- prep progress


def test_ffmpeg_progress_is_reported_and_clamped(monkeypatch):
    class FakeProc:
        returncode = 0

        def __init__(self, lines):
            self.stderr = iter(lines)

        def wait(self):
            return 0

    lines = [
        "out_time_ms=2000000\n",  # 2s of a 4s file -> 0.5
        "out_time_ms=bogus\n",
        "out_time_ms=9000000\n",  # past the end -> clamped to 1.0
        "[silencedetect @ 0x0] silence_start: 1.0\n",
    ]
    monkeypatch.setattr("pipeline.subprocess.Popen", lambda *a, **k: FakeProc(lines))
    seen = []
    log = _run_ffmpeg(["ffmpeg"], duration=4.0, on_progress=seen.append)
    assert seen == [0.5, 1.0]
    assert "silence_start" in log


def test_ffmpeg_progress_needs_a_duration(monkeypatch):
    class FakeProc:
        returncode = 0

        def __init__(self):
            self.stderr = iter(["out_time_ms=1000000\n"])

        def wait(self):
            return 0

    monkeypatch.setattr("pipeline.subprocess.Popen", lambda *a, **k: FakeProc())
    seen = []
    _run_ffmpeg(["ffmpeg"], duration=0.0, on_progress=seen.append)
    assert seen == []


def test_a_stalled_ffmpeg_is_killed_by_the_watchdog(monkeypatch):
    """The regression: a full-file pass had no bound at all, and the progress
    loop is blocked reading a pipe so it could not even check a deadline. A
    truncated or hostile file pinned a request thread forever, and cancelling an
    open does not reach these two calls -- Esc only cancels the remux."""
    killed = threading.Event()

    class FakeProc:
        returncode = -9

        def __init__(self):
            self.stderr = self._log()

        def _log(self):
            while not killed.is_set():
                yield "still going\n"
                time.sleep(0.01)

        def poll(self):
            return -9 if killed.is_set() else None

        def kill(self):
            killed.set()

        def wait(self):
            while not killed.is_set():
                time.sleep(0.01)
            return -9

    monkeypatch.setattr("pipeline.subprocess.Popen", lambda *a, **k: FakeProc())
    start = time.time()
    try:
        _run_ffmpeg(["ffmpeg"], duration=1.0, timeout=0.2)
    except TimeoutError as exc:
        assert "0.2s" in str(exc)
    else:
        raise AssertionError("a stalled ffmpeg was never timed out")
    assert killed.is_set(), "the child was left running"
    assert time.time() - start < 3.0, "the watchdog did not fire promptly"


def test_the_ffmpeg_budget_scales_with_the_media_and_always_bounded():
    # Long media needs minutes; a tiny file still gets a sane ceiling.
    assert _ffmpeg_budget(3600.0) > 3600.0
    assert _ffmpeg_budget(0.0) > 0.0, "an unprobeable duration must still be bounded"
    assert _ffmpeg_budget(1.0) >= 60.0


# --------------------------------------------------------------- planning


def test_plan_respects_max_len_and_covers_duration():
    chunks = plan_chunks(100.0, [29.0], target=30, max_len=30, snap=6)
    assert chunks[0] == (0.0, 29.0)
    assert chunks[-1][1] == 100.0
    assert all(b - a <= 30.0 for a, b in chunks)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == prev_end, "chunks must be contiguous"


def test_plan_snaps_to_nearest_silence():
    chunks = plan_chunks(60.0, [27.0, 29.5], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 29.5


def test_plan_rejects_snap_past_max_len():
    """A silence beyond whisper's 30s window must not stretch the chunk."""
    chunks = plan_chunks(60.0, [33.0], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 30.0


def test_plan_ignores_distant_silence():
    chunks = plan_chunks(60.0, [5.0], target=30, max_len=30, snap=6)
    assert chunks[0][1] == 30.0


def test_plan_short_and_empty():
    assert plan_chunks(0.0, []) == []
    assert plan_chunks(12.0, []) == [(0.0, 12.0)]


# -------------------------------------------------------------- scheduling


def make_chunks(n, length=30.0):
    return [(i * length, (i + 1) * length) for i in range(n)]


def make_scheduler(chunks, log, lookahead=10.0):
    def run_chunk(idx, t0, t1):
        log.append(idx)
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(
        chunks, run_chunk, lookahead=lookahead, poll=0.01
    )
    return scheduler


def wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_fills_to_end():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["finished"])
    scheduler.stop()
    assert log == [0, 1, 2, 3]


def test_transcription_runs_eagerly_ahead_of_the_playhead():
    """Pins the deliberate design: the worker sweeps the whole file regardless of
    where the playhead is. `lookahead` is only the client's highlight threshold,
    not a gate -- see README's "Runs the whole file" note. If this ever fails,
    someone re-added a gate that the docs no longer promise.
    """
    log = []
    scheduler = make_scheduler(make_chunks(6), log)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["finished"])
    scheduler.stop()
    scheduler.set_playhead(150.0)  # parked on the last chunk
    assert log == [0, 1, 2, 3, 4, 5], "every chunk ran before the playhead moved"


def test_a_poisoned_chunk_is_retried_then_skipped_not_ground_on():
    """The regression: a chunk that always throws was never written to `cache`,
    so `_pick()` returned the same index on every pass -- a 1Hz spin that re-took
    the ASR lock each time and starved the synthesiser for the whole session."""
    attempts = []

    def run_chunk(idx, t0, t1):
        attempts.append(idx)
        if idx == 1:
            raise RuntimeError("unrecoverable")
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(make_chunks(4), run_chunk, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["finished"], timeout=8.0)
    scheduler.stop()

    assert attempts.count(1) == LookaheadScheduler._TRIES, \
        "a permanently failing chunk must not retry forever"
    assert sorted(set(attempts)) == [0, 1, 2, 3], "the sweep must continue past it"
    state = scheduler.state()
    assert state["failed"] == [1]
    assert state["cached"] == [0, 1, 2, 3], "the gap is cached empty so it is counted"
    assert state["finished"], "one bad chunk must not strand the session"
    assert 1 not in [c.source for c in scheduler.all_cues()], \
        "a skipped chunk must contribute no cues"


def test_a_skipped_chunks_warning_survives_later_successes():
    """The error is cleared on every success, which would have made the next
    good chunk silently erase the notice that a chunk was dropped."""
    def run_chunk(idx, t0, t1):
        if idx == 0:
            raise RuntimeError("unrecoverable")
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(make_chunks(3), run_chunk, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["finished"], timeout=8.0)
    scheduler.stop()
    assert "chunk 0" in scheduler.state()["error"]
    assert scheduler.state()["error"] is not None


def test_seek_scans_forward_before_backfilling_behind():
    log = []
    hold = True

    def run_chunk(idx, t0, t1):
        log.append(idx)
        while hold and idx == 0:
            time.sleep(0.005)
        return [Cue(t0, t1, f"chunk {idx}")]

    scheduler = LookaheadScheduler(make_chunks(10), run_chunk, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: 0 in log)
    scheduler.set_playhead(95.0)  # jump to chunk 3
    hold = False
    assert wait_until(lambda: scheduler.state()["finished"])
    scheduler.stop()
    assert log[0] == 0
    assert log[log.index(3):log.index(9) + 1] == list(range(3, 10)), \
        "the seek target must stream to the end first"
    assert log.index(1) > log.index(9), "chunks behind are back-filled last"
    assert log.index(2) > log.index(9)
    assert sorted(log) == list(range(10))


def test_seek_back_serves_from_cache():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: 0 in log)
    scheduler.set_playhead(95.0)
    assert wait_until(lambda: scheduler.state()["chunks_done"] >= 2)
    scheduler.set_playhead(5.0)
    time.sleep(0.05)
    scheduler.stop()
    assert log.count(0) == 1, "cached chunks must not be re-run"


def test_warming_clears_after_first_chunk():
    log = []
    scheduler = make_scheduler(make_chunks(3), log)
    assert scheduler.state()["warming"] is True
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["warming"] is False)
    scheduler.stop()
    assert 0 in log


def test_state_reports_ahead():
    log = []
    scheduler = make_scheduler(make_chunks(10), log)
    scheduler.start()
    assert wait_until(lambda: len(log) >= 1)
    scheduler.set_playhead(5.0)
    assert wait_until(lambda: scheduler.state()["ahead"] >= 25.0)
    state = scheduler.state()
    scheduler.stop()
    assert state["transcribed_until"] >= 30.0
    assert state["error"] is None


def test_errors_are_surfaced_not_fatal():
    def boom(idx, t0, t1):
        raise RuntimeError("model exploded")

    scheduler = LookaheadScheduler(make_chunks(3), boom, poll=0.01)
    scheduler.start()
    assert wait_until(lambda: scheduler.state()["error"] is not None)
    scheduler.stop()
    assert "model exploded" in scheduler.state()["error"]


def test_all_cues_are_ordered():
    log = []
    scheduler = make_scheduler(make_chunks(4), log)
    scheduler.start()
    assert wait_until(lambda: len(log) >= 1)
    scheduler.set_playhead(95.0)
    assert wait_until(lambda: len(scheduler.all_cues()) >= 2)
    cues = scheduler.all_cues()
    scheduler.stop()
    assert [c.start for c in cues] == sorted(c.start for c in cues)


# ------------------------------------------------------------------ reflow


def test_reflow_splits_long_cue_into_two_line_pieces():
    text = " ".join(["word"] * 40)
    cues = reflow_cues([Cue(0.0, 9.0, text, text.upper())])
    assert len(cues) >= 2
    for cue in cues:
        for side in (cue.source, cue.target):
            assert side.count("\n") <= 1
            assert all(len(line) <= 44 for line in side.split("\n"))
    assert cues[0].start == 0.0
    assert cues[-1].end == 9.0
    for prev, nxt in zip(cues, cues[1:]):
        assert abs(prev.end - nxt.start) < 1e-9


def test_reflow_leaves_short_cues_alone():
    cues = reflow_cues([Cue(0.0, 2.0, "hello world")])
    assert [(c.start, c.end, c.source) for c in cues] == [(0.0, 2.0, "hello world")]


def test_reflow_handles_cjk_without_spaces():
    cues = reflow_cues([Cue(0.0, 6.0, "あ" * 100)])
    assert len(cues) >= 2
    assert all(len(line) <= 42 for c in cues for line in c.source.split("\n"))


# ------------------------------------------------------- chapter frame grabs


def test_frame_jpeg_serves_from_cache_without_touching_ffmpeg(monkeypatch, tmp_path):
    """Hover previews re-ask for the same chapter, so a cached frame must not
    re-run ffmpeg."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not really a video")
    root = tmp_path / "frames"
    root.mkdir()
    (root / "12500.jpg").write_bytes(b"\xff\xd8already here")

    def explode(*args, **kwargs):
        raise AssertionError("ffmpeg must not run on a cache hit")

    monkeypatch.setattr(subprocess, "run", explode)
    assert frame_jpeg(video, 12.5, root).read_bytes() == b"\xff\xd8already here"


def test_frame_jpeg_extracts_once_and_seeks_before_the_input(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    root = tmp_path / "frames"
    runs = []

    def fake_run(args, **kwargs):
        runs.append(args)
        return SimpleNamespace(stdout=b"\xff\xd8fresh", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert frame_jpeg(video, 12.504, root).read_bytes() == b"\xff\xd8fresh"
    frame_jpeg(video, 12.504, root)  # second hover: cached
    assert len(runs) == 1, runs
    # -ss must precede -i or ffmpeg decodes from the start of the file, which on
    # a two-hour video is the difference between 70ms and minutes.
    assert runs[0].index("-ss") < runs[0].index("-i")
    assert "-frames:v" in runs[0]
    assert runs[0][runs[0].index("-ss") + 1] == "12.504"


def test_frame_jpeg_is_none_when_there_is_no_picture(monkeypatch, tmp_path):
    """Audio-only input: the caller draws a placeholder rather than a broken img."""
    video = tmp_path / "a.m4a"
    video.write_bytes(b"")
    root = tmp_path / "frames"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"", returncode=1)
    )
    assert frame_jpeg(video, 3.0, root) is None


# ------------------------------------------------------------ cache management


def build_cache(root: Path, keys: dict) -> Path:
    """Lay out a cache the way the app does, one entry per key.

    Everything for a key is spread over a flat `<key>.f32`/`.json`/`.mp4`,
    `frames/<key>/`, and `tts/<key>/<voice>/`, so a helper that only looked at
    one shape would badly misreport the real footprint. `age` is seconds ago the
    key was written, which is what eviction sorts on.
    """
    root.mkdir(parents=True, exist_ok=True)
    for key, (size, age) in keys.items():
        (root / f"{key}.f32").write_bytes(b"x" * size)
        (root / f"{key}.json").write_text("{}")
        frames = root / "frames" / key
        frames.mkdir(parents=True, exist_ok=True)
        (frames / "1000.jpg").write_bytes(b"x" * (size // 2))
        voice = root / "tts" / key / "af_heart"
        voice.mkdir(parents=True, exist_ok=True)
        (voice / "2000.wav").write_bytes(b"x" * (size // 4))
        when = time.time() - age
        for path in [root / f"{key}.f32", root / f"{key}.json",
                     frames, voice]:
            os.utime(path, (when, when))
    return root


def test_cache_entries_sums_every_shape_of_one_key(monkeypatch, tmp_path):
    root = build_cache(tmp_path / "c", {"aaaa": (1000, 10)})
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    entries = cache_entries()
    assert list(entries) == ["aaaa"]
    # f32 1000 + json 2 + frames 500 + tts 250
    assert entries["aaaa"][1] == 1752


def test_prune_evicts_the_oldest_key_first(monkeypatch, tmp_path):
    root = build_cache(tmp_path / "c", {
        "oldest": (4000, 500),   # 4000 + 2 + 2000 + 1000 = 7002
        "middle": (4000, 300),
        "newest": (4000, 10),
    })
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    result = prune_cache(max_mb=0.018)  # 18874 bytes: room for one of three
    assert result["evicted"] == ["oldest"]
    assert sorted(p.name for p in root.iterdir() if p.is_file()) == [
        "middle.f32", "middle.json", "newest.f32", "newest.json"]
    assert not (root / "frames" / "oldest").exists(), "a pruned key's frames must go too"
    assert not (root / "tts" / "oldest").exists()
    assert result["freed"] == 7002
    assert result["bytes"] == 2 * 7002  # .f32 4000 + .json 2 + frames 2000 + tts 1000


def test_prune_never_touches_the_live_sessions_key(monkeypatch, tmp_path):
    """The active video's PCM may be an open memmap and its remux may still be
    being written. Unlinking those does not fail loudly -- the mapping survives
    -- so the damage surfaces later as missing cues, not as an error."""
    root = build_cache(tmp_path / "c", {"live": (4000, 500), "stale": (4000, 300)})
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    result = prune_cache(protect={"live"}, max_mb=0.001)  # budget nothing fits
    assert result["evicted"] == ["stale"]
    assert (root / "live.f32").is_file()
    assert (root / "frames" / "live").is_dir()
    assert not (root / "stale.f32").exists()


def test_prune_keeps_everything_that_already_fits(monkeypatch, tmp_path):
    root = build_cache(tmp_path / "c", {"a": (100, 10), "b": (100, 20)})
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    result = prune_cache(max_mb=10)
    assert result["evicted"] == []
    assert result["freed"] == 0
    assert len(cache_entries()) == 2


def test_prune_is_a_no_op_when_the_budget_is_disabled(monkeypatch, tmp_path):
    root = build_cache(tmp_path / "c", {"a": (9000, 10)})
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    result = prune_cache(max_mb=0)
    assert result["evicted"] == []
    assert (root / "a.f32").is_file()


def test_prune_survives_a_cache_dir_that_does_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.CACHE_DIR", tmp_path / "nope")
    assert cache_entries() == {}
    assert prune_cache(max_mb=1)["evicted"] == []
    assert remove_cache_key("aaaa") == 0


def test_a_partial_remux_is_collected_as_its_own_key_group(monkeypatch, tmp_path):
    """A `.part.mp4` left by a killed run belongs to its key, so it is either
    kept with that video or collected with it -- never mistaken for a key of
    its own."""
    root = tmp_path / "c"
    root.mkdir()
    (root / "abcd.part.mp4").write_bytes(b"x" * 500)
    monkeypatch.setattr("pipeline.CACHE_DIR", root)
    assert list(cache_entries()) == ["abcd"]
    remove_cache_key("abcd")
    assert list(root.iterdir()) == []
