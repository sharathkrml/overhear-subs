const $ = (id) => document.getElementById(id);

const el = {
  video: $("video"),
  frame: $("frame"),
  stage: document.querySelector(".stage"),
  controls: $("controls"),
  play: $("play"),
  back10: $("back10"),
  fwd10: $("fwd10"),
  mute: $("mute"),
  volume: $("volume"),
  ctlTime: $("ctl-time"),
  speed: $("speed"),
  speedMenu: $("speed-menu"),
  captions: $("captions"),
  speak: $("speak"),
  voice: $("voice"),
  voiceWrap: $("voice-wrap"),
  pip: $("pip"),
  fullscreen: $("fullscreen"),
  cancel: $("cancel"),
  help: $("help"),
  helpClose: $("help-close"),
  copyAll: $("copy-all"),
  overlay: $("overlay"),
  pulse: $("pulse"),
  fileName: $("file-name"),
  session: $("session"),
  transcript: $("transcript"),
  chapters: $("chapters"),
  chapterList: $("chapter-list"),
  chapterNote: $("chapter-note"),
  chLane: $("ch-lane"),
  chNow: $("ch-now"),
  chNowT: $("ch-now-t"),
  chNowX: $("ch-now-x"),
  chRail: $("ch-rail"),
  chRailTrack: $("ch-rail-track"),
  chaptersBtn: $("chapters-btn"),
  chPreview: $("ch-preview"),
  chPreviewImg: $("ch-preview-img"),
  chPreviewTitle: $("ch-preview-title"),
  chPreviewTime: $("ch-preview-time"),
  panelIntro: $("panel-intro"),
  panelSkeleton: $("panel-skeleton"),
  panelEmpty: $("panel-empty"),
  follow: $("follow"),
  status: $("status"),
  meter: $("meter"),
  meterTrack: document.querySelector(".meter-track"),
  meterCells: $("meter-cells"),
  meterHead: $("meter-head"),
  clock: $("clock"),
  note: $("note"),
  empty: $("empty"),
  reset: $("reset"),
  open: $("open"),
  emptyOpen: $("empty-open"),
  fallback: $("fallback"),
  fallbackPath: $("fallback-path"),
  export: $("export"),
  veil: $("veil"),
  veilFile: $("veil-file"),
  veilTitle: $("veil-title"),
  veilHint: $("veil-hint"),
  veilBar: $("veil-bar"),
  veilFill: $("veil-fill"),
  steps: $("steps"),
  banner: $("banner"),
  bannerText: $("banner-text"),
  save: $("save"),
};

const S = {
  ws: null,
  cues: [],
  keys: new Set(),
  rows: new Map(),
  lookahead: 10,
  duration: 0,
  activeKey: null,
  hint: 0,
  lastSent: 0,
  dragging: false,
  hasMedia: false,
  finished: false,
  busy: false,
  buckets: 0,
  chunksPerBucket: 1,
  pendingVideo: null,
  mediaInfo: null,
  phase: "idle",
  warming: false,
  videoReady: false,
  playback: null,
  prep: null,
  captions: true,
  speak: false,
  tts: null,
  chapters: null,
  chapterRows: new Map(),
  chapterSegs: new Map(),
  chapterCount: 0,
  activeChapter: null,
  chapterModel: "",
  railOpen: false,
  previewTimer: 0,
  speed: 1,
  helpOpen: false,
  retry: 0,
  gen: 0,
};

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];
const clamp01 = (v) => Math.min(1, Math.max(0, v));
let hideTimer = 0;

// ------------------------------------------------------------- transport

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws = ws;

  ws.onopen = () => {
    S.retry = 0;
    if (S.hasMedia) sendPlayhead(true);
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "hello") {
      S.lookahead = msg.lookahead;
      if (!msg.native_picker) showFallback();
      if (msg.chapters) S.chapterModel = msg.chapters.model || "";
      if (msg.media) restore(msg.media);
    } else if (msg.type === "cues") {
      addCues(msg.items);
    } else if (msg.type === "state") {
      renderState(msg);
    }
  };

  ws.onclose = () => {
    setStatus("Disconnected — reconnecting…");
    S.retry += 1;
    setTimeout(connect, Math.min(8000, 500 * S.retry));
  };
}

function sendPlayhead(force) {
  if (!S.ws || S.ws.readyState !== 1) return;
  const now = performance.now();
  if (!force && now - S.lastSent < 250) return;
  S.lastSent = now;
  S.ws.send(JSON.stringify({ type: "playhead", time: el.video.currentTime }));
}

// ------------------------------------------------------------------ cues

function addCues(items) {
  if (!items || !items.length) return;
  const fresh = [];
  for (const cue of items) {
    const key = cue.start.toFixed(2);
    if (S.keys.has(key)) continue;
    S.keys.add(key);
    // Keep the key on the same object the row renders from, or the row's
    // dataset/S.rows entry is keyed by `undefined` and highlighting dies.
    const keyed = { ...cue, key };
    S.cues.push(keyed);
    fresh.push(keyed);
  }
  if (!fresh.length) return;
  S.cues.sort((a, b) => a.start - b.start);
  if (S.speak) ttsSchedule(false);
  fresh.sort((a, b) => a.start - b.start);

  const nearBottom =
    el.transcript.scrollHeight - el.transcript.scrollTop - el.transcript.clientHeight < 90;
  for (const cue of fresh) el.transcript.appendChild(row(cue));
  // A backfill (open / seek) drops a whole batch at once; stagger the first few
  // so it reads as lines arriving, not a flash. Capped so it never feels slow.
  if (fresh.length > 1) {
    fresh.forEach((cue, i) => {
      const li = S.rows.get(cue.key);
      if (li) li.style.animationDelay = `${Math.min(i, 8) * 30}ms`;
    });
  }
  showPanel(null);
  el.export.hidden = false;
  if (nearBottom && el.follow.checked) el.transcript.scrollTop = el.transcript.scrollHeight;
  if (!S.warming) hideBanner();
}

function cueText(cue) {
  return !cue.target || cue.target === cue.source
    ? cue.source
    : `${cue.source}\n${cue.target}`;
}

function flashCopied(btn) {
  const label = btn.textContent;
  btn.textContent = "Copied";
  btn.classList.add("done");
  setTimeout(() => {
    btn.textContent = label;
    btn.classList.remove("done");
  }, 1200);
}

function row(cue) {
  const li = document.createElement("li");
  li.dataset.key = cue.key;
  li.tabIndex = 0;
  li.setAttribute("role", "button");
  li.setAttribute("aria-label", `Seek to ${clock(cue.start)}`);

  const time = document.createElement("span");
  time.className = "t";
  time.textContent = clock(cue.start);

  const body = document.createElement("div");
  body.className = "x";
  const src = document.createElement("div");
  src.className = "src";
  src.textContent = cue.source;
  body.appendChild(src);
  if (cue.target && cue.target !== cue.source) {
    const tgt = document.createElement("div");
    tgt.className = "tgt";
    tgt.textContent = cue.target;
    body.appendChild(tgt);
  }

  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy";
  copy.textContent = "Copy";
  copy.title = "Copy line";
  copy.setAttribute("aria-label", "Copy line");
  copy.onclick = (event) => {
    event.stopPropagation();
    navigator.clipboard.writeText(cueText(cue)).then(
      () => flashCopied(copy),
      () => {}
    );
  };

  li.append(time, body, copy);
  // A clean click seeks; a click that ends a text selection must not.
  li.addEventListener("click", () => {
    if (window.getSelection().toString().trim()) return;
    seekTo(cue.start + 0.01);
  });
  li.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      seekTo(cue.start + 0.01);
    }
  });
  S.rows.set(cue.key, li);
  return li;
}

// -------------------------------------------------------------- chapters

// The server buckets the transcript into windows and has a local model pick the
// cuts inside each one, so chapters arrive a window after you watch them rather
// than all at once at the end. The strip says why it is empty instead of
// showing a blank box.
function renderChapters(state) {
  const c = state.chapters;
  if (!c) return;
  if (c.model) S.chapterModel = c.model;
  S.chapters = c;
  const list = Array.isArray(c.chapters) ? c.chapters : [];
  // The state pump ticks twice a second; only a new chapter justifies a rebuild.
  if (list.length !== S.chapterCount) {
    S.chapterCount = list.length;
    S.chapterRows = new Map();
    el.chapterList.replaceChildren(...list.map(chapterRow));
    buildChapterLane(list);
    buildChapterRail(list);
    S.activeChapter = null;
  }
  const note = chapterNote(c);
  el.chapterNote.textContent = note;
  el.chapterNote.hidden = !note;
  el.chapters.hidden = !S.hasMedia;
}

function chapterNote(state) {
  if (state.error) return state.error;
  if (!state.enabled) {
    return `start ollama and pull ${S.chapterModel || "a model"} to get chapters`;
  }
  if (!S.chapterCount) return `first chapter after ${clock(state.every || 300)}`;
  return "";
}

function chapterRow(ch, i, list) {
  const li = document.createElement("li");
  li.tabIndex = 0;
  // The thumbnail is what tells a chapter apart from a subtitle at a glance.
  const thumb = document.createElement("img");
  thumb.alt = "";
  thumb.loading = "lazy";
  thumb.src = frameURL(ch.start);
  const time = document.createElement("span");
  time.className = "t";
  time.textContent = clock(ch.start);
  const text = document.createElement("span");
  text.className = "x";
  text.textContent = ch.title;
  li.append(thumb, time, text);
  li.addEventListener("click", () => seekTo(ch.start + 0.01));
  li.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      event.stopPropagation();
      seekTo(ch.start + 0.01);
    }
  });
  hoverPreview(li, ch, chapterEnd(list, i));
  S.chapterRows.set(ch.start, li);
  return li;
}

const frameURL = (t) => `/api/frame?start=${t.toFixed(2)}`;

// Where a chapter stops: the next one starts, or the end of the video.
function chapterEnd(list, i) {
  if (i + 1 < list.length) return list[i + 1].start;
  return S.duration || list[i].start;
}

// Chapters on the timeline. Segment widths are proportional to each chapter's
// span, so the shape of the video reads at a glance and every segment is wide
// enough to be a hover target.
function buildChapterLane(list) {
  const duration = S.duration || 0;
  S.chapterSegs = new Map();
  el.chLane.replaceChildren();
  if (!list.length || !duration) {
    el.chLane.hidden = true;
    return;
  }
  for (let i = 0; i < list.length; i++) {
    const start = list[i].start;
    const end = i + 1 < list.length ? list[i + 1].start : duration;
    const seg = document.createElement("div");
    seg.className = "ch-seg";
    seg.style.flexGrow = String(Math.max(1, end - start));
    // A click on a chapter means "go to the chapter", not "scrub to this pixel",
    // so keep it out of the meter's own pointer handling.
    seg.addEventListener("pointerdown", (event) => event.stopPropagation());
    seg.addEventListener("click", (event) => {
      event.stopPropagation();
      seekTo(start + 0.01);
    });
    hoverPreview(seg, list[i], end);
    S.chapterSegs.set(start, seg);
    el.chLane.appendChild(seg);
  }
  el.chLane.hidden = false;
}

// The fullscreen chapter surface. Fullscreen hides the panel *and* the cursor,
// so a hover-only affordance is unreachable there; this is a real list of
// thumbnail cards you open deliberately, and it survives the transport's
// auto-hide because it is navigation rather than transport.
function buildChapterRail(list) {
  S.chapterCards = new Map();
  el.chRailTrack.replaceChildren();
  for (let i = 0; i < list.length; i++) {
    const start = list[i].start;
    const card = document.createElement("button");
    card.type = "button";
    card.className = "ch-card";
    const img = document.createElement("img");
    img.alt = "";
    img.loading = "lazy";
    img.src = frameURL(start);
    const meta = document.createElement("span");
    meta.className = "ch-card-meta";
    const time = document.createElement("b");
    time.textContent = clock(start);
    const text = document.createElement("span");
    text.textContent = list[i].title;
    meta.append(time, text);
    card.append(img, meta);
    card.setAttribute("aria-label", `Seek to ${clock(start)}, ${list[i].title}`);
    card.addEventListener("click", () => seekTo(start + 0.01));
    S.chapterCards.set(start, card);
    el.chRailTrack.appendChild(card);
  }
  el.chaptersBtn.hidden = !list.length;
  setRail(S.railOpen && list.length > 0);
}

function setRail(open) {
  S.railOpen = open && S.chapterCount > 0;
  el.chRail.hidden = !S.railOpen;
  el.chaptersBtn.setAttribute("aria-pressed", String(S.railOpen));
  el.chaptersBtn.title = S.railOpen ? "Hide chapters" : "Chapters";
  if (S.railOpen) S.chapterCards.get(S.activeChapter)?.scrollIntoView({ block: "nearest", inline: "center" });
}

el.chaptersBtn.addEventListener("click", () => setRail(!S.railOpen));

// Hovering either surface (timeline segment or panel row) shows the same card:
// the frame the chapter starts on, its title, and how far it runs. Rail cards
// already show their own frame, so they opt out.
function hoverPreview(node, ch, end) {
  node.addEventListener("pointerenter", () => showPreview(node, ch, end));
  node.addEventListener("pointerleave", scheduleHidePreview);
  node.addEventListener("focus", () => showPreview(node, ch, end));
  node.addEventListener("blur", scheduleHidePreview);
}

function showPreview(node, ch, end) {
  clearTimeout(S.previewTimer);
  if (!ch) return hidePreview();
  const stop = end ?? S.duration ?? ch.start;
  el.chPreviewTitle.textContent = ch.title;
  el.chPreviewTime.textContent = `${clock(ch.start)} – ${clock(stop)}`;
  // The img is only fetched once a hover asks for it, and /api/frame is
  // immutable per time, so re-hovering is a cache hit.
  if (el.chPreviewImg.dataset.start !== String(ch.start)) {
    el.chPreviewImg.dataset.start = String(ch.start);
    el.chPreviewImg.src = frameURL(ch.start);
  }
  el.chPreview.hidden = false;

  // Absolute inside .frame, so measure against the frame's own box and keep the
  // card inside it on both axes.
  const host = el.frame.getBoundingClientRect();
  const box = el.chPreview.getBoundingClientRect();
  const from = node.getBoundingClientRect();
  const pad = 10;
  const left = Math.max(pad, Math.min(
    from.left - host.left + from.width / 2 - box.width / 2,
    host.width - box.width - pad
  ));
  // Lane segments sit low in the frame, so the card goes above them; a panel row
  // is off to the side, so the card centres beside it instead.
  const above = from.top < host.top + host.height / 2;
  const top = above
    ? from.top - host.top - box.height - 10
    : from.top - host.top + (from.height - box.height) / 2;
  el.chPreview.style.left = `${left}px`;
  el.chPreview.style.top = `${Math.max(pad, Math.min(top, host.height - box.height - pad))}px`;
}

function scheduleHidePreview() {
  clearTimeout(S.previewTimer);
  S.previewTimer = setTimeout(hidePreview, 120);
}

function hidePreview() {
  el.chPreview.hidden = true;
  el.chPreviewImg.removeAttribute("src");
  delete el.chPreviewImg.dataset.start;
}

function clearChapters() {
  S.chapters = null;
  S.chapterRows = new Map();
  S.chapterSegs = new Map();
  S.chapterCards = new Map();
  S.chapterCount = 0;
  S.activeChapter = null;
  S.railOpen = false;
  clearTimeout(S.previewTimer);
  hidePreview();
  el.chLane.replaceChildren();
  el.chLane.hidden = true;
  el.chRailTrack.replaceChildren();
  el.chRail.hidden = true;
  el.chaptersBtn.hidden = true;
  el.chaptersBtn.setAttribute("aria-pressed", "false");
  el.chNow.hidden = true;
  el.chNowT.textContent = "";
  el.chNowX.textContent = "";
  el.chapterList.replaceChildren();
  el.chapterNote.textContent = "";
  el.chapterNote.hidden = true;
  el.chapters.hidden = true;
}

// The chapter covering `t` is the last one that starts before it. Guarded by the
// cached key so a 60fps tick only touches the DOM when the chapter changes.
function setActiveChapter(t) {
  let hit = null;
  for (const ch of S.chapters?.chapters || []) {
    if (ch.start > t + 0.25) break;
    hit = ch;
  }
  const key = hit ? hit.start : null;
  if (key === S.activeChapter) return;
  clearActive(S.activeChapter);
  S.activeChapter = key;
  if (hit) {
    el.chNow.hidden = false;
    el.chNowT.textContent = clock(hit.start);
    el.chNowX.textContent = hit.title;
  } else {
    el.chNow.hidden = true;
  }
  paintActive(key, S.chapterRows);
  paintActive(key, S.chapterSegs);
  paintActive(key, S.chapterCards);
}

function paintActive(key, map) {
  if (key !== null && key !== undefined) map?.get(key)?.classList.add("active");
}

function clearActive(key) {
  if (key === null || key === undefined) return;
  S.chapterRows.get(key)?.classList.remove("active");
  S.chapterSegs.get(key)?.classList.remove("active");
  S.chapterCards.get(key)?.classList.remove("active");
}

function backfill() {
  fetch("/api/cues")
    .then((r) => r.json())
    .then((d) => addCues(d.items))
    .catch(() => {});
}

function refreshState() {
  fetch("/api/state")
    .then((r) => r.json())
    .then((s) => {
      if (s && s.chunks_total !== undefined) renderState(s);
    })
    .catch(() => {});
}

function clearCues() {
  S.cues = [];
  S.keys = new Set();
  S.rows = new Map();
  S.activeKey = null;
  S.hint = 0;
  S.buckets = 0;
  S.finished = false;
  el.transcript.replaceChildren();
  el.meterCells.replaceChildren();
  clearChapters();
  showPanel(null);
  el.export.hidden = true;
  el.save.hidden = true;
  el.overlay.textContent = "";
  el.overlay.classList.remove("show");
}

function findCue(t) {
  const cues = S.cues;
  for (let i = Math.max(0, S.hint - 2); i < cues.length; i++) {
    if (cues[i].start > t) return -1;
    if (t <= cues[i].end + 0.6) return i;
  }
  return -1;
}

// -------------------------------------------------------------- playback

function seekTo(time) {
  const t = Math.max(0, Math.min(S.duration || el.video.duration || 0, time));
  if (S.dragging && el.video.fastSeek) el.video.fastSeek(t);
  else el.video.currentTime = t;
  paintHead(t);
  if (!S.dragging) sendPlayhead(true);
}

function paintHead(t) {
  const d = S.duration || 0;
  const pct = d ? Math.min(100, Math.max(0, (t / d) * 100)) : 0;
  el.meterHead.style.left = `${pct}%`;
  el.meterHead.style.opacity = d ? "1" : "0";
  el.meter.setAttribute("aria-valuenow", Math.round(pct));
  el.meter.setAttribute("aria-valuetext", clock(t));
  el.clock.textContent = `${clock(t)} / ${clock(d)}`;
  el.ctlTime.textContent = `${clock(t)} / ${clock(d)}`;
}

// ----------------------------------------------------------------- meter

function timelineSeek(event) {
  const rect = el.meterTrack.getBoundingClientRect();
  const pct = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  seekTo(pct * (S.duration || 0));
}

el.meter.addEventListener("pointerdown", (event) => {
  if (!S.duration) return;
  S.dragging = true;
  el.meter.dataset.active = "true";
  el.meter.setPointerCapture(event.pointerId);
  timelineSeek(event);
});

el.meter.addEventListener("pointermove", (event) => {
  if (S.dragging) timelineSeek(event);
});

const endDrag = (event) => {
  if (!S.dragging) return;
  S.dragging = false;
  el.meter.dataset.active = "false";
  if (el.meter.hasPointerCapture(event.pointerId)) el.meter.releasePointerCapture(event.pointerId);
  sendPlayhead(true);
};
el.meter.addEventListener("pointerup", endDrag);
el.meter.addEventListener("pointercancel", endDrag);

// ------------------------------------------------------------- rendering

function tick() {
  const t = el.video.currentTime;
  const idx = findCue(t);
  if (idx >= 0 && S.captions) {
    S.hint = idx;
    const cue = S.cues[idx];
    el.overlay.textContent = cue.target || cue.source;
    el.overlay.classList.add("show");
    setActive(cue.key);
  } else if (idx >= 0) {
    S.hint = idx;
    el.overlay.classList.remove("show");
    setActive(cue.key);
  } else {
    el.overlay.classList.remove("show");
    setActive(null);
  }
  if (!S.dragging) paintHead(t);
  if (S.chapterCount) setActiveChapter(t);
  if (!el.video.paused) sendPlayhead(false);
  requestAnimationFrame(tick);
}

function setActive(key) {
  if (key === S.activeKey) return;
  if (S.activeKey && S.rows.has(S.activeKey)) S.rows.get(S.activeKey).classList.remove("active");
  S.activeKey = key;
  if (key && S.rows.has(key)) {
    const li = S.rows.get(key);
    li.classList.add("active");
    if (el.follow.checked) li.scrollIntoView({ block: "nearest" });
  }
}

function renderState(state) {
  if (state.playback) S.playback = state.playback;
  if (state.prep) S.prep = state.prep;
  if (state.tts) renderTTS(state);
  if (state.chapters) renderChapters(state);
  if (!S.hasMedia) {
    el.pulse.classList.remove("live", "done");
    if (S.phase === "opening") {
      paintPhase();
    } else if (S.phase === "idle") {
      el.note.textContent = "No video open";
      el.note.classList.remove("ahead");
    }
    return;
  }
  S.finished = !!state.finished;
  S.warming = !!state.warming;

  paintCells(state);

  if (S.pendingVideo && S.playback && S.playback.ready) {
    if (S.playback.error) {
      setStatus(`Couldn't convert for playback: ${S.playback.error}`, true);
    }
    attachVideo(S.pendingVideo, S.mediaInfo);
  }

  el.pulse.classList.toggle("live", !S.finished);
  el.pulse.classList.toggle("done", !!S.finished);

  paintPhase();
  el.save.hidden = !S.finished || !el.veil.hidden;
  if (!el.veil.hidden) return;

  const ahead = state.ahead ?? 0;
  const done = state.chunks_done ?? 0;
  const total = state.chunks_total ?? 0;
  const working = !S.finished && !S.warming && !state.error;
  el.note.classList.toggle("ahead", (S.finished || ahead >= S.lookahead) && !S.warming);
  el.note.classList.toggle("behind", working && ahead < 0);
  if (state.error) {
    el.note.textContent = "Transcription error";
    setStatus(state.error, true);
  } else if (S.warming) {
    el.note.textContent = "Loading model…";
  } else {
    // The headline number: how much transcript is already saved past the
    // playhead, measured live — not a fixed window.
    const lead = ahead >= 0
      ? `+${ahead.toFixed(1)}s transcribed`
      : `${Math.abs(ahead).toFixed(1)}s behind`;
    el.note.textContent = total ? `${lead} · ${done}/${total}` : lead;
  }
}

// One cell per chunk. Above ~120 chunks we bucket them so the DOM and the
// gaps stay sane; a bucket lights as soon as any chunk in it lands.
function buildCells(total) {
  S.buckets = Math.min(total, 120);
  S.chunksPerBucket = total / S.buckets;
  el.meterCells.replaceChildren();
  for (let i = 0; i < S.buckets; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    el.meterCells.appendChild(cell);
  }
}

function paintCells(state) {
  if (!S.buckets) return;
  const done = new Set();
  for (const idx of state.cached || []) {
    done.add(Math.min(S.buckets - 1, Math.floor(idx / S.chunksPerBucket)));
  }
  const cells = el.meterCells.children;
  for (let i = 0; i < cells.length; i++) cells[i].classList.toggle("done", done.has(i));
}

function setStatus(text, isError) {
  el.status.textContent = text || "";
  el.status.classList.toggle("error", !!isError);
}

function clock(t) {
  const total = Math.max(0, Math.floor(t || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? String(m).padStart(2, "0") : m;
  return `${h ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`;
}

function showVeil({ title, file, hint, steps, bar, cancel } = {}) {
  el.empty.hidden = true;
  el.veil.hidden = false;
  el.controls.hidden = true;
  el.frame.classList.remove("controls-on");
  el.cancel.hidden = !cancel;
  el.veilTitle.textContent = title || "Preparing";
  el.veilFile.textContent = file || "";
  el.veilFile.hidden = !file;
  el.veilHint.textContent = hint || "";
  el.veilHint.hidden = !hint;
  el.steps.hidden = !steps;
  if (steps) {
    for (const li of el.steps.children) {
      li.className = steps[li.dataset.step] || "";
    }
  }
  if (bar == null) {
    el.veilBar.hidden = true;
  } else {
    el.veilBar.hidden = false;
    el.veilFill.style.width = `${Math.round(Math.min(1, Math.max(0, bar)) * 100)}%`;
  }
}

function hideVeil() {
  el.veil.hidden = true;
  el.controls.hidden = !S.hasMedia;
  if (!S.hasMedia) el.empty.hidden = false;
}

// The right panel is never blank: intro before a video, skeleton while its
// transcript is being prepared, text for a status, nothing once cues exist.
function showPanel(mode) {
  el.panelIntro.hidden = mode !== "intro";
  el.panelSkeleton.hidden = mode !== "loading";
  el.panelEmpty.hidden = mode !== "text";
}

function showBanner(text) {
  el.bannerText.textContent = text;
  el.banner.hidden = false;
}

function hideBanner() {
  el.banner.hidden = true;
}

function paintPhase() {
  const converting =
    S.pendingVideo && S.playback && !S.playback.ready && !S.playback.error;
  const file = el.fileName.textContent;

  if (S.phase === "idle") {
    hideVeil();
    hideBanner();
    el.save.hidden = true;
    el.note.textContent = "No video open";
    el.note.classList.remove("ahead", "behind");
    showPanel("intro");
    el.panelEmpty.textContent = "Open a video to start.";
    return;
  }

  if (S.phase === "picking") {
    showVeil({
      title: "Choose a video",
      hint: "The file panel should be in front of the browser",
    });
    el.note.textContent = "Waiting for a file…";
    showPanel("text");
    el.panelEmpty.textContent = "Waiting for a video…";
    return;
  }

  if (S.phase === "opening") {
    const prep = S.prep;
    const pct = prep ? Math.round(clamp01(prep.progress || 0) * 100) : 0;
    showVeil({
      title: "Opening",
      file,
      hint: prep
        ? `${prep.label}… ${pct}% (Esc to cancel)`
        : "Extracting audio and planning chunks… (Esc to cancel)",
      steps: { audio: "active", play: "", model: "" },
      bar: prep ? clamp01(prep.progress || 0) : null,
      cancel: true,
    });
    el.note.textContent = prep ? `${prep.label}…` : "Opening…";
    showPanel("loading");
    el.panelEmpty.textContent = prep ? `${prep.label}…` : "Extracting audio…";
    return;
  }

  if (converting) {
    const pct = Math.round((S.playback.progress || 0) * 100);
    const why = S.playback.reason || "video";
    showVeil({
      title: "Preparing playback",
      file,
      hint: `Converting ${why} so the browser can play it… ${pct}% (Esc to cancel)`,
      steps: { audio: "done", play: "active", model: S.warming ? "active" : "" },
      bar: S.playback.progress || 0,
      cancel: true,
    });
    el.note.textContent = `Converting… ${pct}%`;
    showPanel("loading");
    el.panelEmpty.textContent = "Preparing playback…";
    return;
  }

  if (S.hasMedia && !S.videoReady) {
    showVeil({
      title: "Loading video",
      file,
      hint: S.warming ? "Loading Auto → English (first run downloads ~3 GB)…" : "",
      steps: { audio: "done", play: "active", model: S.warming ? "active" : "" },
    });
    el.note.textContent = "Loading video…";
    return;
  }

  hideVeil();

  if (S.hasMedia && S.warming) {
    showBanner("Loading Auto → English… first run downloads ~3 GB");
  } else {
    hideBanner();
  }
  if (!S.cues.length) showPanel("loading");
}

// -------------------------------------------------------------- transport

function showControls() {
  if (!S.hasMedia) return;
  el.frame.classList.add("controls-on");
  clearTimeout(hideTimer);
  if (!el.video.paused) {
    hideTimer = setTimeout(() => el.frame.classList.remove("controls-on"), 2600);
  }
}

function hideControls() {
  clearTimeout(hideTimer);
  if (!el.video.paused) el.frame.classList.remove("controls-on");
}

// --------------------------------------------------------- read-along (TTS)

// The synthesiser writes each cue to disk ~10s ahead of the playhead, so this
// side only has to keep a decoded buffer per cue and hand it to Web Audio at the
// right moment. Web Audio rather than a second <audio> element because its
// scheduling is sample-accurate, which is what keeps the dub locked to the
// video across seeks, stalls and 0.5x-2x playback.
const TTS_HORIZON = 15; // cue seconds to schedule ahead of the playhead
const TTS_DUCK = 0.15;  // original-audio level while the translation speaks

const A = {
  ctx: null,
  gen: 0,             // bump to void in-flight decodes after a voice change
  buffers: new Map(),  // ms -> AudioBuffer
  pending: new Map(),  // ms -> Promise, so a cue is never fetched twice
  live: new Set(),     // sources currently sounding
  scheduled: new Set(),
  voice: null,         // voice the decoded buffers belong to
};

const duck = { active: false, base: 1 };

function actx() {
  if (!A.ctx) A.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (A.ctx.state === "suspended") A.ctx.resume();
  return A.ctx;
}

const cueMs = (cue) => Math.round(cue.start * 1000);

function ttsBuffer(cue, gen) {
  const ms = cueMs(cue);
  if (A.buffers.has(ms)) return Promise.resolve(A.buffers.get(ms));
  if (A.pending.has(ms)) return A.pending.get(ms);
  const p = fetch(`/api/tts/audio?start=${cue.start}`)
    .then((r) => (r.ok ? r.arrayBuffer() : null))
    .then((buf) => (buf ? actx().decodeAudioData(buf) : null))
    .then((decoded) => {
      if (!decoded || gen !== A.gen) return null; // voice changed mid-decode
      A.buffers.set(ms, decoded);
      return decoded;
    })
    .catch(() => null)
    .finally(() => A.pending.delete(ms));
  A.pending.set(ms, p);
  return p;
}

function ttsPlay(cue, buffer, at, head, rate) {
  const ms = cueMs(cue);
  if (A.scheduled.has(ms)) return;
  A.scheduled.add(ms);
  // Seeked into the middle of a line: skip the part already spoken instead of
  // restarting the whole cue from its first word.
  const skip = Math.max(0, (head - cue.start) * rate);
  if (skip >= buffer.duration) return;
  const src = actx().createBufferSource();
  src.buffer = buffer;
  src.connect(actx().destination);
  src.start(at + Math.max(0, (cue.start - head) / rate), skip);
  A.live.add(src);
  src.onended = () => A.live.delete(src);
}

function ttsStop() {
  for (const src of A.live) {
    try {
      src.stop();
    } catch {}
  }
  A.live.clear();
  A.scheduled.clear();
}

function ttsSchedule(reset) {
  if (reset) ttsStop();
  if (!S.speak || !S.hasMedia) return;
  const head = el.video.currentTime;
  const rate = el.video.playbackRate || 1;
  for (const cue of S.cues) {
    if (cue.start > head + TTS_HORIZON) break;
    if (cue.end < head) continue; // already past
    const have = A.buffers.get(cueMs(cue));
    if (have) {
      ttsPlay(cue, have, actx().currentTime, head, rate);
    } else {
      // 404 just means the synthesiser hasn't reached this cue yet; the next
      // pass retries it, which is how audio fills in as it lands.
      ttsBuffer(cue, A.gen).then((buf) => {
        if (buf && S.speak) {
          ttsPlay(cue, buf, actx().currentTime, el.video.currentTime,
                  el.video.playbackRate || 1);
        }
      });
    }
  }
}

// Full re-anchor: the playhead moved, so everything must be re-timed.
const ttsAnchor = () => ttsSchedule(true);

function ttsReset() {
  ttsStop();
  A.gen += 1;
  A.buffers.clear();
  A.pending.clear();
}

function applyDuck() {
  duck.active = S.speak;
  el.video.volume = duck.active ? duck.base * TTS_DUCK : duck.base;
  syncVolume();
}

function renderTTS(state) {
  const t = state.tts;
  if (!t) return;
  const was = S.speak;
  const grew = (S.tts?.spoken ?? 0) < (t.spoken ?? 0);
  S.tts = t;
  S.speak = !!t.enabled;
  el.speak.setAttribute("aria-pressed", String(S.speak));
  el.voiceWrap.hidden = !S.speak;
  el.speak.dataset.phase = t.error ? "error" : t.enabled && !t.ready ? "loading" : "on";
  el.speak.title = t.error
    ? `Read-aloud failed: ${t.error}`
    : t.enabled && !t.ready
      ? "Loading the voice model…"
      : "Read the translation aloud (R)";

  if (S.speak && el.voice.options.length !== (t.voices || []).length) {
    el.voice.replaceChildren(
      ...t.voices.map((v) => new Option(v.replace(/^[a-z]{2}_/, "").replace(/_/g, " "), v))
    );
  }
  if (t.voice) el.voice.value = t.voice;

  if (A.voice && t.voice && t.voice !== A.voice) {
    ttsReset(); // a different voice caches under a different directory
  }
  A.voice = t.voice;

  if (was !== S.speak) {
    if (S.speak) {
      duck.base = el.video.muted ? duck.base : el.video.volume;
      actx(); // unlock audio on the gesture that turned it on
    } else {
      ttsReset();
    }
    applyDuck();
  }
  // Extend rather than re-anchor: a re-anchor would restart whatever is
  // mid-sentence every tick. New audio only ever needs adding.
  if (S.speak && (was !== S.speak || grew)) ttsSchedule(was !== S.speak);
}

async function setSpeak(on) {
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on, voice: el.voice.value || undefined }),
    });
    if (!res.ok) throw new Error(await res.text());
    renderTTS({ tts: await res.json() });
  } catch (e) {
    setStatus(`Read-aloud: ${e.message}`, true);
  }
}

function syncVolume() {
  const level = el.video.muted ? 0 : duck.active ? duck.base : el.video.volume;
  const muted = el.video.muted || level === 0;
  el.mute.classList.toggle("muted", muted);
  el.volume.value = String(level);
  el.volume.setAttribute("aria-valuetext", `${Math.round(level * 100)}%`);
}

function setVolume(v) {
  el.video.muted = false;
  duck.base = clamp01(Number(v));
  applyDuck();
}

function toggleMute() {
  el.video.muted = !el.video.muted;
  if (!el.video.muted && duck.base === 0) duck.base = 0.5;
  applyDuck();
}

function setSpeed(rate) {
  S.speed = rate;
  el.video.playbackRate = rate;
  el.speed.textContent = `${rate}×`;
  for (const item of el.speedMenu.children) {
    item.setAttribute("aria-checked", String(Number(item.dataset.speed) === rate));
  }
}

function stepSpeed(dir) {
  const i = SPEEDS.indexOf(S.speed);
  const j = Math.min(SPEEDS.length - 1, Math.max(0, (i < 0 ? 2 : i) + dir));
  if (j !== i) setSpeed(SPEEDS[j]);
}

function openSpeedMenu() {
  el.speedMenu.hidden = false;
  el.speed.setAttribute("aria-expanded", "true");
}

function closeSpeedMenu() {
  el.speedMenu.hidden = true;
  el.speed.setAttribute("aria-expanded", "false");
}

function toggleCaptions() {
  S.captions = !S.captions;
  el.captions.setAttribute("aria-pressed", String(S.captions));
  if (!S.captions) el.overlay.classList.remove("show");
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await el.stage.requestFullscreen();
  } catch {}
}

async function togglePiP() {
  try {
    if (document.pictureInPictureElement) await document.exitPictureInPicture();
    else if (el.video.requestPictureInPicture) await el.video.requestPictureInPicture();
  } catch {}
}

function openHelp() {
  S.helpOpen = true;
  el.help.hidden = false;
  el.helpClose.focus();
}

function closeHelp() {
  S.helpOpen = false;
  el.help.hidden = true;
}

async function cancelOpen() {
  if (S.busy) return;
  S.gen += 1; // any open still in flight must not install its result
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch {}
  resetSession();
  setStatus("Cancelled");
}

// ------------------------------------------------------------------ open

function showFallback() {
  el.open.hidden = true;
  el.emptyOpen.hidden = true;
  el.fallback.hidden = false;
  $("empty-hint").textContent = "Paste a path in the field above · local files only";
  el.fallbackPath.focus();
}

async function pick() {
  if (S.busy) return;
  if (el.fallback.hidden === false) {
    return open(el.fallbackPath.value.trim());
  }
  S.busy = true;
  S.phase = "picking";
  paintPhase();
  try {
    const res = await fetch("/api/pick", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const { path } = await res.json();
    if (path) await open(path);
    else {
      S.phase = S.hasMedia ? "playing" : "idle";
      setStatus("");
      paintPhase();
    }
  } catch (err) {
    S.phase = S.hasMedia ? "playing" : "idle";
    setStatus(String(err.message || err), true);
    paintPhase();
  } finally {
    S.busy = false;
  }
}

async function open(path) {
  if (!path) return;
  const gen = ++S.gen;
  S.phase = "opening";
  S.prep = null;
  el.fileName.textContent = path.split("/").pop();
  el.fileName.title = path;
  paintPhase();
  el.open.disabled = true;
  el.emptyOpen.disabled = true;
  try {
    const res = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(await res.text());
    const info = await res.json();
    if (gen !== S.gen) return; // cancelled or superseded while opening
    applyMedia(path, info, `/media?t=${Date.now()}`);
  } catch (err) {
    if (gen !== S.gen) return;
    S.phase = S.hasMedia ? "playing" : "idle";
    setStatus(String(err.message || err), true);
    paintPhase();
  } finally {
    if (gen === S.gen) {
      el.open.disabled = false;
      el.emptyOpen.disabled = false;
    }
  }
}

function restore(media) {
  applyMedia(media.path, media, "/media");
}

function resetSession() {
  clearCues();
  ttsReset();
  S.hasMedia = false;
  S.duration = 0;
  S.mediaInfo = null;
  S.playback = null;
  S.prep = null;
  S.videoReady = false;
  S.warming = false;
  S.pendingVideo = null;
  S.phase = "idle";
  el.video.pause();
  el.video.removeAttribute("src");
  el.video.load();
  el.frame.classList.remove("controls-on");
  el.controls.hidden = true;
  el.session.hidden = true;
  el.reset.hidden = true;
  el.open.hidden = true;
  setStatus("");
  paintPhase();
}

async function resetAndPick() {
  if (S.busy) return;
  S.gen += 1;
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch {}
  resetSession();
  if (el.fallback.hidden === false) {
    el.fallbackPath.value = "";
    el.fallbackPath.focus();
  } else {
    pick();
  }
}

function applyMedia(path, info, url) {
  clearCues();
  S.hasMedia = true;
  S.duration = info.duration;
  S.mediaInfo = info;
  S.playback = info.playback || { ready: true };
  S.videoReady = false;
  S.warming = true;
  S.phase = "playing";
  buildCells(info.chunks || 1);
  el.fileName.textContent = path.split("/").pop();
  el.fileName.title = path;
  el.session.hidden = false;
  if (el.fallback.hidden) el.open.hidden = false;
  el.reset.hidden = false;
  el.empty.hidden = true;
  el.video.removeAttribute("src");
  backfill();
  refreshState();

  const playback = S.playback;
  if (playback.ready) {
    if (playback.error) {
      setStatus(`Couldn't convert for playback: ${playback.error}`, true);
    }
    attachVideo(url, info);
  } else {
    S.pendingVideo = url;
    paintPhase();
  }
}

function attachVideo(url, info) {
  S.pendingVideo = null;
  S.videoReady = false;
  el.video.src = url;
  el.video.load();
  setStatus(
    `${info.chunks} chunks · ${clock(info.duration)}` +
      (info.playback && info.playback.converted ? " · converted for playback" : "")
  );
  paintPhase();
  showControls();
}

// ----------------------------------------------------------------- wiring

el.open.onclick = pick;
el.reset.onclick = resetAndPick;
el.emptyOpen.onclick = pick;
el.fallback.onsubmit = (event) => {
  event.preventDefault();
  open(el.fallbackPath.value.trim());
};

for (const btn of document.querySelectorAll(".seg button")) {
  btn.onclick = () => (location.href = `/api/export?fmt=${btn.dataset.fmt}`);
}

// -- transport controls ----------------------------------------------------

for (const rate of SPEEDS) {
  const item = document.createElement("button");
  item.type = "button";
  item.setAttribute("role", "menuitemradio");
  item.dataset.speed = String(rate);
  item.textContent = `${rate}×`;
  item.onclick = () => {
    setSpeed(rate);
    closeSpeedMenu();
  };
  el.speedMenu.appendChild(item);
}

el.play.onclick = () => (el.video.paused ? el.video.play() : el.video.pause());
el.back10.onclick = () => seekTo(el.video.currentTime - 10);
el.fwd10.onclick = () => seekTo(el.video.currentTime + 10);
el.mute.onclick = toggleMute;
el.volume.oninput = () => setVolume(el.volume.value);
el.captions.onclick = toggleCaptions;
el.speak.onclick = () => setSpeak(!S.speak);
el.voice.onchange = () => setSpeak(S.speak);
el.pip.onclick = togglePiP;
el.fullscreen.onclick = toggleFullscreen;
el.cancel.onclick = cancelOpen;
el.helpClose.onclick = closeHelp;
el.speed.onclick = (event) => {
  event.stopPropagation();
  el.speedMenu.hidden ? openSpeedMenu() : closeSpeedMenu();
};
el.help.addEventListener("click", (event) => {
  if (event.target === el.help) closeHelp();
});
document.addEventListener("click", closeSpeedMenu);
el.copyAll.onclick = async () => {
  try {
    const res = await fetch("/api/export?fmt=txt");
    await navigator.clipboard.writeText(await res.text());
    flashCopied(el.copyAll);
  } catch {}
};

el.frame.addEventListener("mousemove", showControls);
el.frame.addEventListener("mouseleave", hideControls);

el.video.addEventListener("click", () => {
  if (!S.hasMedia || !el.video.src) return;
  el.video.paused ? el.video.play() : el.video.pause();
});
el.video.addEventListener("dblclick", () => {
  if (S.hasMedia && el.video.src) toggleFullscreen();
});
if (!document.pictureInPictureEnabled) el.pip.hidden = true;

el.video.addEventListener("play", () => {
  el.play.classList.add("playing");
  el.play.setAttribute("aria-label", "Pause");
  showControls();
  actx();
  ttsAnchor();
});
el.video.addEventListener("pause", () => {
  el.play.classList.remove("playing");
  el.play.setAttribute("aria-label", "Play");
  showControls();
  ttsStop(); // don't keep speaking over a paused frame
});

document.addEventListener("fullscreenchange", () => {
  const on = !!document.fullscreenElement;
  el.fullscreen.setAttribute("aria-label", on ? "Exit fullscreen" : "Fullscreen");
  el.fullscreen.title = on ? "Exit fullscreen (F)" : "Fullscreen (F)";
  if (on) showControls();
});

syncVolume();
setSpeed(S.speed);

el.video.addEventListener("loadedmetadata", () => {
  if (el.video.duration && isFinite(el.video.duration)) S.duration = el.video.duration;
  paintHead(el.video.currentTime);
  sendPlayhead(true);
});
el.video.addEventListener("canplay", () => {
  S.videoReady = true;
  paintPhase();
});
el.video.addEventListener("seeked", () => {
  S.hint = 0;
  sendPlayhead(true);
  backfill();
  ttsAnchor();
});
el.video.addEventListener("ratechange", ttsAnchor);

const MEDIA_ERRORS = {
  1: "loading was aborted",
  2: "a network error occurred",
  3: "the video could not be decoded",
  4: "this format isn't supported by the browser",
};
el.video.addEventListener("error", () => {
  if (!el.video.error) return;
  S.videoReady = true;
  setStatus(
    `Playback failed: ${MEDIA_ERRORS[el.video.error.code] || el.video.error.message}`,
    true
  );
  paintPhase();
});

// One router for every shortcut. Typing fields keep their keys; buttons keep
// Space and Enter. Everything else plays nicely with the player.
document.addEventListener("keydown", (event) => {
  const active = document.activeElement;
  const tag = (active && active.tagName) || "";
  const typing = /INPUT|TEXTAREA|SELECT/.test(tag) || !!(active && active.isContentEditable);
  const key = event.key;
  const lower = key.toLowerCase();

  if ((event.metaKey || event.ctrlKey) && lower === "o") {
    event.preventDefault();
    pick();
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey) return;

  if (key === "?" && !typing) {
    event.preventDefault();
    S.helpOpen ? closeHelp() : openHelp();
    return;
  }
  if (key === "Escape") {
    if (S.helpOpen) return closeHelp();
    if (!el.speedMenu.hidden) return closeSpeedMenu();
    if (typing) return;
    if (document.fullscreenElement) return;
    if (S.phase === "opening" || (S.pendingVideo && S.playback && !S.playback.ready && !S.playback.error)) {
      event.preventDefault();
      cancelOpen();
    }
    return;
  }
  if (typing) return;

  if (key === " " || lower === "k") {
    if (/BUTTON|A/.test(tag) || !S.hasMedia || !el.video.src) return;
    event.preventDefault();
    el.video.paused ? el.video.play() : el.video.pause();
    return;
  }
  if (!S.hasMedia || !el.video.src) return;

  const step = event.shiftKey ? 30 : 5;
  switch (key) {
    case "ArrowLeft":
      event.preventDefault();
      seekTo(el.video.currentTime - step);
      break;
    case "ArrowRight":
      event.preventDefault();
      seekTo(el.video.currentTime + step);
      break;
    case "ArrowUp":
      event.preventDefault();
      setVolume(el.video.muted ? 0.05 : el.video.volume + 0.05);
      break;
    case "ArrowDown":
      event.preventDefault();
      setVolume(el.video.volume - 0.05);
      break;
    case "Home":
      event.preventDefault();
      seekTo(0);
      break;
    case "End":
      event.preventDefault();
      seekTo(S.duration);
      break;
    case ",":
      if (el.video.paused) {
        event.preventDefault();
        seekTo(el.video.currentTime - 1 / 30);
      }
      break;
    case ".":
      if (el.video.paused) {
        event.preventDefault();
        seekTo(el.video.currentTime + 1 / 30);
      }
      break;
    case "<":
      event.preventDefault();
      stepSpeed(-1);
      break;
    case ">":
      event.preventDefault();
      stepSpeed(1);
      break;
    default:
      if (lower === "j") {
        event.preventDefault();
        seekTo(el.video.currentTime - 10);
      } else if (lower === "l") {
        event.preventDefault();
        seekTo(el.video.currentTime + 10);
      } else if (lower === "m") {
        event.preventDefault();
        toggleMute();
      } else if (lower === "f") {
        event.preventDefault();
        toggleFullscreen();
      } else if (lower === "c") {
        event.preventDefault();
        toggleCaptions();
      } else if (lower === "p") {
        event.preventDefault();
        togglePiP();
      } else if (lower === "r") {
        event.preventDefault();
        if (S.hasMedia) setSpeak(!S.speak);
      } else if (/^[0-9]$/.test(key)) {
        event.preventDefault();
        seekTo((Number(key) / 10) * S.duration);
      }
  }
});

connect();
requestAnimationFrame(tick);
