"""FastAPI app: file picking, media serving, look-ahead scheduling, export."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import backends
from pipeline import (
    CACHE_DIR,
    LOOKAHEAD,
    VIDEO_EXT,
    LookaheadScheduler,
    PlaybackPrep,
    TTSSynthesizer,
    _cache_key,
    build_media,
)

STATIC = Path(__file__).parent / "static"

# Off by default: the model is a ~350 MB one-time download, and a session that
# never asks for read-along shouldn't pay for it.
TTS_ON = os.environ.get("LT_TTS", "0") not in ("0", "", "false", "no")

log = logging.getLogger("overhear-subs")


def _warm_backend() -> None:
    """Load the ASR weights at boot, not on the first chunk of a video."""
    try:
        backends.get_backend().warm()
    except Exception:
        log.exception("model warm-up failed; it will load lazily on first use")


def _warm_tts() -> None:
    """Load Kokoro + espeak off the request thread, then backfill every cue
    already transcribed so enabling mid-playback isn't silent for a chunk."""
    try:
        backends.get_tts().warm()
    except Exception:
        log.exception("TTS warm-up failed; it will retry on the next cue")
    tts, scheduler = session.tts, session.scheduler
    if tts and tts.enabled and scheduler:
        tts.submit(scheduler.all_cues())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Backgrounded so the server can answer before a ~3 GB first-run download.
    threading.Thread(target=_warm_backend, daemon=True, name="warm-backend").start()
    yield


app = FastAPI(title="overhear-subs", lifespan=lifespan)


@app.middleware("http")
async def _no_store_static(request, call_next):
    """Keep the browser from serving stale app.js/style.css after an edit."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-store"
    return response


NATIVE_PICKER = sys.platform == "darwin" and shutil.which("osascript") is not None
_PICK_LOCK = threading.Lock()

_VIDEO_TYPES = "{" + ", ".join(f'"{ext.lstrip(".")}"' for ext in sorted(VIDEO_EXT)) + "}"

_CHOOSE_FILE = f"""try
    tell application "System Events" to activate
end try
try
    set theFile to choose file with prompt "Choose a video to transcribe" default location (path to movies folder) of type {_VIDEO_TYPES}
    return POSIX path of theFile
on error number -128
    return ""
end try"""


def choose_file() -> str | None:
    """Native macOS open panel. Returns None if the user cancels."""
    args: list[str] = ["osascript"]
    for line in _CHOOSE_FILE.splitlines():
        args += ["-e", line]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise HTTPException(500, (proc.stderr or "osascript failed").strip())
    return proc.stdout.strip() or None



class Prep:
    """Media-prep progress for the opening veil: "audio" then "silence"."""

    _LABELS = {"audio": "Extracting audio", "silence": "Planning chunks"}

    def __init__(self) -> None:
        self.stage = "audio"
        self.progress = 0.0

    def update(self, stage: str, frac: float) -> None:
        base = 0.0 if stage == "audio" else 0.5
        self.stage = stage
        self.progress = base + max(0.0, min(1.0, frac)) * 0.5

    def state(self) -> dict:
        return {"label": self._LABELS.get(self.stage, "Preparing"),
                "progress": round(self.progress, 3)}


class Session:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.playback: PlaybackPrep | None = None
        self.scheduler: LookaheadScheduler | None = None
        self.tts: TTSSynthesizer | None = None
        self.prep: Prep | None = None
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None


session = Session()

# Bumped by every open/reset. A slow open that finishes after a reset must not
# install its session — the user cancelled it.
_open_gen = 0


class OpenReq(BaseModel):
    path: str


class SeekReq(BaseModel):
    time: float


# --------------------------------------------------------------------------
# websocket plumbing
# --------------------------------------------------------------------------


def _state_payload() -> dict:
    payload: dict = {}
    if session.prep:
        payload["prep"] = session.prep.state()
    if session.playback:
        payload["playback"] = session.playback.state()
    if session.scheduler:
        payload.update(session.scheduler.state())
    if session.tts:
        payload["tts"] = session.tts.state()
    return payload


def _tts_state() -> dict:
    """Shape the client needs even before a media is open: is the feature
    available, is the model resident yet, and what can be picked."""
    state = session.tts.state() if session.tts else {
        "enabled": False, "ready": False, "spoken": 0, "queued": 0, "error": None,
    }
    state.update(model=backends.TTS_MODEL, voice=backends.get_tts().voice,
                 voices=list(backends.KokoroTTS.voices),
                 sample_rate=backends.KokoroTTS.sample_rate)
    return state


async def broadcast(message: dict) -> None:
    dead = []
    for client in list(session.clients):
        try:
            await client.send_json(message)
        except Exception:
            dead.append(client)
    for client in dead:
        session.clients.discard(client)


def _on_cues(idx: int, cues: list) -> None:
    # Feed the synthesiser from the scheduler's own callback: a chunk landing is
    # exactly the moment its cues become ten seconds of runway from the playhead.
    if session.tts:
        session.tts.submit(cues)
    if not session.loop or not session.clients:
        return
    payload = {"type": "cues", "items": [c.__dict__ for c in cues]}
    asyncio.run_coroutine_threadsafe(broadcast(payload), session.loop)


async def _state_pump(client: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(0.5)
            await client.send_json({"type": "state", **_state_payload()})
    except Exception:
        pass


@app.websocket("/ws")
async def websocket(client: WebSocket) -> None:
    await client.accept()
    session.clients.add(client)
    session.loop = asyncio.get_running_loop()
    pump = asyncio.create_task(_state_pump(client))
    try:
        await client.send_json(
            {
                "type": "hello",
                "lookahead": LOOKAHEAD,
                "native_picker": NATIVE_PICKER,
                "open": session.path is not None,
                "tts": _tts_state(),
                "duration": (
                    session.scheduler.chunks[-1][1]
                    if session.scheduler and session.scheduler.chunks
                    else 0.0
                ),
                "media": (
                    {
                        "path": str(session.path),
                        "duration": session.scheduler.chunks[-1][1],
                        "chunks": len(session.scheduler.chunks),
                        "playback": session.playback.state() if session.playback else None,
                    }
                    if session.path and session.scheduler
                    else None
                ),
            }
        )
        while True:
            msg = await client.receive_json()
            if msg.get("type") == "playhead" and session.scheduler:
                session.scheduler.set_playhead(float(msg["time"]))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        pump.cancel()
        session.clients.discard(client)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/pick")
def pick() -> dict:
    """Open the native file panel. Blocks until the user chooses or cancels."""
    if not NATIVE_PICKER:
        raise HTTPException(501, "native file picker unavailable on this platform")
    if not _PICK_LOCK.acquire(blocking=False):
        raise HTTPException(409, "a file panel is already open")
    try:
        return {"path": choose_file()}
    finally:
        _PICK_LOCK.release()


@app.post("/api/open")
def open_media(req: OpenReq) -> dict:
    global _open_gen
    path = Path(os.path.expanduser(req.path)).resolve()
    if not path.is_file():
        raise HTTPException(404, f"not found: {path}")

    _teardown()
    _open_gen += 1
    gen = _open_gen
    # Kick the (possibly slow) conversion off first so it overlaps audio prep.
    playback = PlaybackPrep(path)
    playback.start()
    prep = Prep()
    if gen == _open_gen:
        session.playback = playback
        session.prep = prep
    # build_media blocks this worker thread, but the event loop keeps pumping
    # websocket state, so the veil sees prep progress while it runs.
    try:
        source, duration, chunks = build_media(path, prep.update)
    except Exception:
        playback.cancel()
        if gen == _open_gen:
            session.playback = None
            session.prep = None
        raise
    if gen != _open_gen:
        playback.cancel()
        raise HTTPException(409, "open cancelled")
    session.prep = None
    backend = backends.get_backend()

    def run_chunk(idx: int, t0: float, t1: float):
        return backend.run(source.slice(t0, t1), t0)

    scheduler = LookaheadScheduler(chunks, run_chunk, on_cues=_on_cues)
    tts_model = backends.get_tts()
    tts = TTSSynthesizer(
        tts_model.speak,
        backends.KokoroTTS.sample_rate,
        CACHE_DIR / "tts" / _cache_key(path),
        tts_model.voice,
        enabled=TTS_ON,
    )
    tts.start()
    session.path = path
    session.playback = playback
    session.scheduler = scheduler
    session.tts = tts
    scheduler.start()
    if TTS_ON:
        _warm_tts()

    return {
        "path": str(path),
        "duration": duration,
        "chunks": len(chunks),
        "lookahead": LOOKAHEAD,
        "url": "/media",
        "playback": playback.state(),
    }


@app.get("/media")
def media() -> FileResponse:
    playback = session.playback
    if not playback:
        raise HTTPException(404, "no media open")
    if not playback.ready:
        raise HTTPException(503, "playback is still being prepared")
    # On conversion failure fall back to the original: it may not play, but
    # transcription still works and the UI can say what went wrong.
    return FileResponse(playback.output or playback.source)


class TTSReq(BaseModel):
    on: bool
    voice: str | None = None


@app.post("/api/tts")
def tts_toggle(req: TTSReq) -> dict:
    """Turn read-along on or off, or change voice. Enabling loads the model in
    the background; the client polls /api/state until `tts.ready` flips."""
    tts = session.tts
    if tts is None:
        raise HTTPException(400, "no media open")
    if req.voice and req.voice in backends.KokoroTTS.voices:
        # Model first, then the cache directory: set_voice re-queues immediately,
        # so the worker must already be synthesising the new voice.
        backends.get_tts().voice = req.voice
        tts.set_voice(req.voice)
    if req.on and not tts.enabled:
        tts.enabled = True
        threading.Thread(target=_warm_tts, daemon=True, name="warm-tts").start()
    elif not req.on:
        tts.enabled = False
    return _tts_state()


@app.get("/api/tts/audio")
def tts_audio(start: float) -> FileResponse:
    """One spoken cue as wav. 404 until the synthesiser has reached it."""
    tts = session.tts
    if tts is None:
        raise HTTPException(404, "no media open")
    path = tts.path_for(start)
    if path is None:
        raise HTTPException(404, "not spoken yet")
    return FileResponse(path, media_type="audio/wav",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/cues")
def cues() -> dict:
    if not session.scheduler:
        return {"items": []}
    return {"items": [c.__dict__ for c in session.scheduler.all_cues()]}


@app.get("/api/state")
def state() -> dict:
    """Immediate snapshot so the meter isn't dead until the first WS tick."""
    return _state_payload()


@app.post("/api/seek")
def seek(req: SeekReq) -> dict:
    if session.scheduler:
        session.scheduler.set_playhead(req.time)
    return {"ok": True}


@app.post("/api/reset")
def reset() -> dict:
    """Drop the current session so the next open starts fresh."""
    global _open_gen
    _open_gen += 1
    _teardown()
    session.path = None
    session.playback = None
    return {"ok": True}


@app.get("/api/export")
def export(fmt: str = "srt") -> PlainTextResponse:
    if not session.scheduler:
        raise HTTPException(400, "no media open")
    items = session.scheduler.all_cues()
    if fmt == "txt":
        body = "\n".join(c.source if not c.target else f"{c.source}\n{c.target}" for c in items)
        return PlainTextResponse(body, headers=_disposition("transcript.txt"))
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for i, cue in enumerate(items, 1):
            lines += [str(i), f"{_ts(cue.start, '.')} --> {_ts(cue.end, '.')}", _text(cue), ""]
        return PlainTextResponse("\n".join(lines), media_type="text/vtt",
                                 headers=_disposition("transcript.vtt"))
    lines = []
    for i, cue in enumerate(items, 1):
        lines += [str(i), f"{_ts(cue.start)} --> {_ts(cue.end)}", _text(cue), ""]
    return PlainTextResponse("\n".join(lines),
                             media_type="application/x-subrip",
                             headers=_disposition("transcript.srt"))


def _text(cue) -> str:
    if cue.target and cue.target != cue.source:
        return f"{cue.source}\n{cue.target}"
    return cue.source or cue.target


def _ts(t: float, sep: str = ",") -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _disposition(name: str) -> dict:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


def _teardown() -> None:
    if session.playback:
        session.playback.cancel()
        session.playback = None
    if session.tts:
        session.tts.stop()
        session.tts = None
    if session.scheduler:
        session.scheduler.stop()
        session.scheduler = None
    session.prep = None
    session.path = None


app.mount("/static", StaticFiles(directory=STATIC), name="static")
