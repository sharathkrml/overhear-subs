<div align="center">

# overhear-subs

**Subtitles that pull up before you do.**

Play a local video — the words are already on screen when the scene reaches them.

[![platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://github.com/sharathkrml/overhear-subs)
[![python](https://img.shields.io/badge/python-3.12-blue?logo=python)](pyproject.toml)
[![offline](https://img.shields.io/badge/network-one--time%20model%20download-success)](#quickstart)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## The idea

> Transcribe what's **about to play**, before you get there.

```mermaid
flowchart LR
    A([playhead]) --> B["captions already here"]
    A -. "upcoming chunks" .-> C["transcribing next chunk"]
    C -. "idle when not needed" .-> D["queued"]
```

An 88-second clip is cut into 3 chunks. At `playhead = 0`, **one** is transcribed. The worker idles until the video catches up.

| Playhead | Scheduler |
| --- | --- |
| Playing | keeps the chunk you're about to hit in the queue |
| Paused | freezes the window, GPU rests |
| Rewound | re-serves from cache, never re-runs the model |

Everything stays on your machine. The only network hit is the one-time model download from Hugging Face.

---

## Pipeline

```mermaid
flowchart TB
    V[local video] --> F1[ffmpeg demux]
    V --> F2[ffmpeg silencedetect]

    F1 --> PCM["mono 16 kHz f32 PCM<br/>(cached · np.memmap)"]
    F2 --> CH["~30s chunks<br/>edges snap to silence"]

    PCM --> S{{LookaheadScheduler<br/>one worker thread}}
    CH --> S
    S --> W[mlx-whisper large-v3<br/>auto-detect + translate]
    W --> R[reflow cues<br/>≤ 2 lines · 42 chars]
    R --> UI["captions + transcript panel<br/>SRT · VTT · TXT"]
    R -. "same 10s runway" .-> V["kokoro-82M read-along<br/>→ per-cue wav"]
```

<details>
<summary><b>What each stage does</b></summary>

| Stage | Detail |
| --- | --- |
| **Demux** | `ffmpeg` → mono 16 kHz float32 PCM, cached by `path + size + mtime`. Loaded as `np.memmap`, so seeking is just `SAMPLE_RATE * seconds` — an array index. No decode-on-the-fly, no VAD. |
| **Chunking** | `plan_chunks` cuts ~30s windows; `silencedetect` (≥0.4s below −35 dB) nudges each edge up to ±6s so words never split mid-vowel. |
| **Scheduler** | One thread, one rule: transcribe the chunk the playhead is inside, plus the ones coming up soon after it, then rest. The chunk you're about to hit jumps the queue; the rest fills in progressively. |
| **Whisper** | `mlx-community/whisper-large-v3-mlx` on the Apple GPU. Built-in `task="translate"` renders any spoken language as English in a single pass. Full `large-v3`, not turbo — turbo silently ignores translation. |
| **Reflow** | `reflow_cues` splits long segments into ≤ 2 balanced lines (≤ 42 chars), re-timed proportionally. CJK hard-wraps. No 3rd line covering the actor's face. |

**Unplayable files.** The browser decides what plays and the extension lies. `probe_codecs` serves browser-safe files untouched; otherwise `PlaybackPrep` transcodes once to H.264/AAC in a background thread (hardware `h264_videotoolbox`) with a live progress bar. Failure isn't fatal — the original is served and the reason shown.

</details>

---

## Quickstart

Requires **macOS on Apple Silicon**, plus `ffmpeg` and `uv`. For read-along, `espeak-ng`.

```sh
brew install ffmpeg uv        # + espeak-ng if you want read-along
make setup        # install deps
make run          # http://localhost:8000
```

Open <http://localhost:8000> → **Open Video…**. That's a native macOS panel, not a web upload — nothing is copied, nothing leaves the machine. First run downloads the model (~3 GB, one time); after that it's resident and offline.

---

## Controls

```text
play/pause · −10s · +10s · volume+mute · time · speed · CC · PiP · fullscreen
```

| Keys | Action | Keys | Action |
| --- | --- | --- | --- |
| `⌘O` | Open a video | `M` | Mute |
| `Space` / `K` | Play / pause | `0`–`9` | Jump to 0–90% |
| `←` / `→` | Seek 5s (`⇧` = 30s) | `Home` / `End` | Start / end |
| `J` / `L` | Seek 10s | `,` / `.` | Frame step (paused) |
| `↑` / `↓` | Volume | `<` / `>` | Playback speed |
| `C` | Toggle captions | `F` | Fullscreen |
| `R` | Read the translation aloud | | |
| `P` | Picture-in-picture | `?` | Shortcut list |
| `Esc` | Cancel in-progress open | | |

Drag the timeline to scrub — it doubles as a pipeline meter, one cell per chunk. Click a transcript line to seek; **Copy** a line on hover or the whole transcript from the header.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | — | Hugging Face token (also `HUGGING_FACE_HUB_TOKEN`) |
| `LT_LOOKAHEAD` | `10.0` | minimum transcript runway (seconds) the worker keeps before pausing |
| `LT_CHUNK` | `30.0` | chunk length in seconds; shorter means a seek waits less for captions |
| `LT_TRANSLATE_MODEL` | `mlx-community/whisper-large-v3-mlx` | ASR repo (must be translate-capable) |
| `LT_REMUX` | `1` | convert unplayable files to browser-safe mp4 (`0` disables) |
| `LT_TTS` | `0` | read-along on at open (`1` enables; the **Speak** button works either way) |
| `LT_TTS_MODEL` | `mlx-community/Kokoro-82M-4bit` | read-along voice repo |
| `LT_TTS_VOICE` | `af_heart` | default voice (28 in the picker) |
| `LT_TTS_LANG` | `a` | Kokoro language code, `a` = American English |
| `LT_CACHE` | `~/.cache/overhear-subs` | derived PCM + remuxed mp4 |
| `PORT` | `8000` | server port (`make run` / `make dev`) |

Set them inline, or copy `.env.example` to `.env` — the Makefile includes it and
exports whatever's set, so `make run`, `make dev`, and `make test` all see it.
Only `make` loads `.env`; running `uvicorn` directly won't (use `uvicorn --env-file .env`).

```sh
HF_TOKEN=hf_xxx LT_LOOKAHEAD=15 make run
```

**One language profile:** Auto → English, single pass. Swap the model via `LT_TRANSLATE_MODEL`; `large-v3-turbo` won't work (it ignores `task="translate"`).

---

## Read-along

Press **Speak** (or `R`) and the English translation is spoken over the video, with the
original audio ducked to 15%. Pick a voice from the dropdown; each voice caches
separately, so switching back is instant.

It rides the same look-ahead the transcriber already keeps. A cue is spoken the moment its
chunk lands — about ten seconds before the playhead reaches it — so playback is instant
and gapless. `Kokoro-82M-4bit` measured **12.7× realtime** on an M1 Pro, which puts the
slowest of eight test cues at 0.38s: 3.8% of the runway. A cue that would overrun its own
subtitle window is re-spoken once, faster, so a fast talker doesn't get clipped.

Scheduling is Web Audio against `video.currentTime`, so the dub stays locked through seeks,
stalls, and 0.5×–2× playback. Seeking into the middle of a line resumes mid-sentence
instead of restarting the cue.

```mermaid
flowchart LR
  A[chunk transcribed] --> B[speak queue]
  B --> C["cache: tts/&lt;video&gt;/&lt;voice&gt;/&lt;ms&gt;.wav"]
  C --> D[browser fetches + decodes]
  D --> E["Web Audio, sample-accurate<br/>at cue.start"]
```

The wavs are the export interface: a dubbed track is `ffmpeg -i video -i <concatenated> -c copy`
against that same directory, with no re-synthesis.

**Needs `espeak-ng`** (`brew install espeak-ng`) — every small open English TTS needs a
grapheme-to-phoneme step, and Kokoro's is misaki's. The app finds Homebrew's copy on both
Apple Silicon and Intel prefixes; override with `PHONEMIZER_ESPEAK_LIBRARY` and
`PHONEMIZER_ESPEAK_DATA_PATH` if it's somewhere else.

---

## Limits (the honest part)

- **How far ahead is approximate.** The worker keeps transcribing until the upcoming chunks are covered, so the runway follows chunk boundaries (~30s, `LT_CHUNK`) rather than landing on an exact number of seconds.
- **One video at a time.** `/media` serves the current file; opening another cancels what's in flight.
- **Cached conversions are re-probed** — a stale cache is discarded, not served.
- **File panel uses `osascript`** (`app.py:choose_file`). macOS may ask once to control System Events (only to bring the panel forward); denied = panel may open behind the browser. Non-macOS falls back to a path field.
- **Chunk planning re-runs `silencedetect`** on every open (fast, audio-only); PCM extraction is cached.
- **Read-along is English-only.** Whisper translates to English, and Kokoro is strongest in
  English, so speaking the translation is the coherent path. Other Kokoro languages need
  `LT_TTS_LANG` and a matching Whisper target language.
- **Read-along needs espeak-ng**, one more system dependency than the rest of the app.
- **Cues can overlap.** A line spoken faster than `TTS_MAX_SPEED` (1.6×) is allowed to bleed
  into the next one rather than be cut off mid-word.
- **One shared GPU.** Whisper and Kokoro hold weights in the same unified memory. A seek
  saturates the ASR and TTS goes briefly late; the queue absorbs it, the audio just arrives
  later.
- **Settings are stateless** — volume, speed, captions, read-along and sidebar reset per
  session. No localStorage.

---

## Design & tests

UI follows Apple's fluid-interface guidance: feedback on pointer-*down*, 1:1 timeline tracking via `setPointerCapture`, translucent `backdrop-filter` chrome, tabular numerals on clocks. Honors `prefers-reduced-motion`, `prefers-reduced-transparency`, `prefers-contrast`, and `prefers-color-scheme`. Transcript rows are keyboard-reachable, transport fully labeled, every control has a focus ring.

```sh
make test    # chunk planning, scheduler, reflow, format detection — fake backend, no model/ffmpeg
make dev     # auto-reload        make run PORT=9000
make check   # byte-compile      make clean / make cache-clean
```

All Python goes through `uv` — no manual venv activation.

---

## License

[MIT](LICENSE) — do whatever, just keep the copyright notice.
