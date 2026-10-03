"""No-model tests for chaptering: window sizing, gating, dedup, clamping.

The LLM is the only boundary that matters here, so `summarize` is faked the same
way `test_tts.py` fakes `speak`.
"""

import math
import time

from pipeline import (
    CHAPTER_AIM,
    CHAPTER_MAX,
    CHAPTER_MIN,
    Chapterer,
    Cue,
    chapter_script,
    chapter_window,
)


def check(label: str, cond: bool, extra: object = "") -> None:
    assert cond, f"{label}: {extra}"


def wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def fake_summarize(reply, calls=None):
    """Records what it was asked for and replays a canned set of cuts."""
    def summarize(text, prev=None):
        if calls is not None:
            calls.append((text, prev))
        return [dict(c) for c in reply]
    return summarize


def cues(start, end, step=5.0):
    """Cues tiling [start, end). The last one runs to `end` so a window whose
    boundary it reaches counts as fully transcribed."""
    out, t = [], start
    while t < end - step:
        out.append(Cue(t, t + step, f"line at {t:.0f}"))
        t += step
    out.append(Cue(max(t, start), end, f"line at {max(t, start):.0f}"))
    return out


def build(reply, every=100.0, finished=False, calls=None):
    ch = Chapterer(fake_summarize(reply, calls), every, is_finished=lambda: finished)
    ch.start()
    return ch


# ------------------------------------------------------------------ window size


def test_window_math_tiles_the_video():
    # The aim divides evenly at these lengths.
    check("20 min", chapter_window(1200) == 300.0, chapter_window(1200))
    check("2 h", chapter_window(7200) == 300.0, chapter_window(7200))
    # 3599s / 300 rounds up to 12 windows, so the window shrinks to fit exactly.
    check("1 h", math.isclose(3599 / chapter_window(3599), 12.0), chapter_window(3599))
    for duration in (90.0, 301.0, 400.0, 1234.0, 3599.0, 7200.0):
        every = chapter_window(duration)
        check(f"{duration}s clamped", CHAPTER_MIN <= every <= CHAPTER_MAX, every)
        # Never fewer windows than the aim asks for, which would be one chapter
        # covering the whole thing.
        check(f"{duration}s window count",
              math.ceil(duration / every) <= math.ceil(duration / CHAPTER_AIM),
              (duration, every))
    # A video shorter than the floor still gets one chapter rather than zero.
    check("short video", chapter_window(90.0) == CHAPTER_MIN, chapter_window(90.0))
    # Unknown duration falls back to the aim rather than dividing by zero.
    check("zero duration", chapter_window(0) == CHAPTER_AIM, chapter_window(0))


def test_window_clamps_when_the_aim_is_retuned():
    # An aim above the ceiling must not produce windows longer than the ceiling.
    check("clamped high", chapter_window(100000.0, aim=900.0, lo=180.0, hi=480.0) == 480.0)
    # An aim below the floor must not produce windows shorter than the floor.
    check("clamped low", chapter_window(600.0, aim=20.0, lo=180.0, hi=480.0) == 180.0)


# ------------------------------------------------------------------- prompt text


def test_script_stamps_are_relative_and_whitespace_is_flattened():
    text = chapter_script([Cue(610.0, 615.0, "first\n  line"), Cue(615.0, 620.0, "second")])
    check("relative stamp", text.startswith("[10:10.0] first line"), text)
    check("second line", "\n[10:15.0] second" in text, text)


def test_script_prefers_the_translation_when_there_is_one():
    text = chapter_script([Cue(0.0, 5.0, "original", "translated")])
    check("target used", "[00:00.0] translated" in text, text)


# ------------------------------------------------------------------ gating + dedup


def test_no_call_until_a_whole_window_is_transcribed():
    calls = []
    ch = build([{"start": 0, "title": "Opening"}], every=100.0, calls=calls)
    ch.submit(cues(0.0, 60.0))  # a window's worth minus a minute
    time.sleep(0.2)
    ch.stop()
    check("no premature call", calls == [], calls)


def test_chapter_lands_when_the_window_fills():
    calls = []
    ch = build([{"start": 0, "title": "Opening"}], every=100.0, calls=calls)
    ch.submit(cues(0.0, 100.0))
    check("one chapter", wait_until(lambda: ch.state()["chapters"]))
    ch.stop()
    check("titled", ch.state()["chapters"] == [{"start": 0.0, "title": "Opening"}],
          ch.state()["chapters"])
    check("one call", len(calls) == 1, calls)
    check("no previous on the first", calls[0][1] is None, calls[0][1])


def test_relative_starts_become_absolute_and_the_window_edge_is_anchored():
    calls = []
    # A model that opens on a later line and cuts past the transcript it was given.
    reply = [
        {"start": 42.0, "title": "Chunking"},
        {"start": 90.0, "title": "Evaluation"},
        {"start": 4000.0, "title": "Never mind"},
    ]
    ch = build(reply, every=100.0, calls=calls)
    ch.submit(cues(200.0, 300.0))  # window 2
    check("chapters made", wait_until(lambda: len(ch.state()["chapters"]) == 3))
    ch.stop()
    starts = [c["start"] for c in ch.state()["chapters"]]
    check("anchored to the window", starts[0] == 200.0, starts)
    check("offset applied", starts[1] == 290.0, starts)
    check("clamped to the transcript", starts[2] == 300.0, starts)


def test_duplicate_cuts_collapse():
    # Two cuts that clamp onto the same instant must not become two rows.
    ch = build([{"start": 10.0, "title": "A"}, {"start": 10.0, "title": "B"}], every=100.0)
    ch.submit(cues(0.0, 100.0))
    check("collapsed", wait_until(lambda: ch.state()["chapters"]))
    ch.stop()
    check("one row", len(ch.state()["chapters"]) == 1, ch.state()["chapters"])


def test_resubmitted_cues_do_not_duplicate_chapters():
    calls = []
    ch = build([{"start": 0, "title": "Opening"}], every=100.0, calls=calls)
    for _ in range(3):  # a backfill sweep re-offers the same cues
        ch.submit(cues(0.0, 100.0))
    check("chapter made", wait_until(lambda: ch.state()["chapters"]))
    time.sleep(0.2)
    ch.stop()
    check("still one", len(ch.state()["chapters"]) == 1, ch.state()["chapters"])
    check("still one call", len(calls) == 1, calls)


def test_previous_title_is_carried_into_the_next_window():
    calls = []
    ch = build([{"start": 0, "title": "Opening"}], every=100.0, calls=calls)
    ch.submit(cues(0.0, 100.0))
    check("first window", wait_until(lambda: len(calls) == 1))
    ch.submit(cues(100.0, 200.0))
    check("second window", wait_until(lambda: len(calls) == 2))
    ch.stop()
    check("seen as previous", calls[1][1] == "Opening", calls[1][1])


def test_an_empty_reply_still_covers_the_window():
    ch = build([], every=100.0)
    ch.submit(cues(0.0, 100.0))
    check("chapter made", wait_until(lambda: ch.state()["chapters"]))
    ch.stop()
    chapters = ch.state()["chapters"]
    check("anchored", chapters[0]["start"] == 0.0, chapters)
    check("titled", bool(chapters[0]["title"]), chapters)


def test_disabled_chapterer_never_calls():
    calls = []
    ch = Chapterer(fake_summarize([], calls), 100.0, enabled=False)
    ch.start()
    ch.submit(cues(0.0, 500.0))
    time.sleep(0.2)
    ch.stop()
    check("no calls", calls == [], calls)


# ------------------------------------------------------------------- the tail


def test_trailing_partial_window_needs_the_transcript_to_be_finished():
    calls = []
    finished = False
    ch = Chapterer(fake_summarize([{"start": 0, "title": "Final"}], calls), 100.0,
                   is_finished=lambda: finished)
    ch.start()
    ch.submit(cues(0.0, 40.0))  # a 100s window that never filled
    time.sleep(0.2)
    check("held back", calls == [], calls)
    finished = True  # the scheduler reports every chunk transcribed
    check("flushed", wait_until(lambda: ch.state()["chapters"]))
    ch.stop()
    check("titled", ch.state()["chapters"] == [{"start": 0.0, "title": "Final"}],
          ch.state()["chapters"])


def test_a_failed_window_retries_then_gives_up():
    attempts = []

    def boom(text, prev=None):
        attempts.append(text)
        raise RuntimeError("weights failed to load")

    ch = Chapterer(boom, 100.0)
    ch.start()
    ch.submit(cues(0.0, 100.0))
    check("error surfaced", wait_until(lambda: ch.error is not None))
    check("retried", wait_until(lambda: len(attempts) >= Chapterer._TRIES, timeout=8.0),
          len(attempts))
    ch.stop()
    check("gave up", ch.state()["chapters"] == [], ch.state()["chapters"])


def test_state_reports_the_window_the_ui_needs():
    ch = Chapterer(fake_summarize([]), 100.0)
    state = ch.state()
    check("every exposed", state["every"] == 100.0, state)
    check("enabled", state["enabled"] is True, state)
    check("empty", state["chapters"] == [], state)


def test_state_exposes_exactly_what_the_client_destructures():
    """app.js renderChapters reads these five keys off the *inner* payload.

    The state pump nests it as payload["chapters"], so a client that forgets to
    destructure sees the wrapper object, .length is undefined, and the chapter
    list silently renders nothing forever. Pin the key set so the wire half of
    that contract cannot drift.
    """
    state = Chapterer(fake_summarize([{"start": 0, "title": "Opening"}]), 100.0).state()
    check("key set", set(state) == {"enabled", "every", "chapters", "error"}, sorted(state))