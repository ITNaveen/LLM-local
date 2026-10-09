/* Live Translator - browser UI (no build step, no external libraries). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const S = {
  status: null,
  settings: null,
  live: { meeting: null, lines: new Map(), els: new Map() },
  view: null,             // {meta, lines, els}
  follow: true,
  ws: null,
  wsDelay: 500,
  meetings: [],
  models: null,
  timer: null,
  startedAt: null,
};

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch (_) { /* empty */ }
  if (!r.ok) throw new Error((data && (data.error || data.detail)) || r.statusText);
  return data;
}

function toast(level, text, ms) {
  const box = $("#toasts");
  // collapse duplicates
  for (const t of $$(".toast", box)) if (t.dataset.text === text) return;
  const el = document.createElement("div");
  el.className = "toast " + (level === "error" ? "error" : level === "warn" ? "warn" : level === "ok" ? "ok" : "");
  el.dataset.text = text;
  el.textContent = text;
  const x = document.createElement("button");
  x.textContent = "✕";
  x.setAttribute("aria-label", "Dismiss");
  x.onclick = () => el.remove();
  el.appendChild(x);
  box.appendChild(el);
  const life = ms !== undefined ? ms : level === "error" ? 0 : 6000;
  if (life) setTimeout(() => el.remove(), life);
}

function fmtDate(iso, opts) {
  try { return new Date(iso).toLocaleString(undefined, { hour12: false, ...opts }); } catch (_) { return iso; }
}
function fmtDur(s) {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}
function dayLabel(dateStr) {
  const d = new Date(dateStr + "T12:00:00");
  const today = new Date(); today.setHours(12, 0, 0, 0);
  const diff = Math.round((today - d) / 86400000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Yesterday";
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" });
}

// ------------------------------------------------------------------ rendering lines
function lineEl(line) {
  const el = document.createElement("article");
  el.className = "line";
  el.dataset.id = line.id;
  el.innerHTML = '<div class="t"></div><div class="body"><div class="de"></div><div class="en"></div></div>';
  fillLine(el, line);
  return el;
}

function fillLine(el, line, meetingId) {
  $(".t", el).textContent = line.time || "";
  $(".de", el).textContent = line.de || "";
  el.classList.toggle("cont", !!line.cont);
  const en = $(".en", el);
  en.className = "en";
  en.textContent = "";
  if (line.en) {
    en.textContent = line.en;
    if (!line.done) en.classList.add("pending");
  } else if (line.done && !line.ok) {
    en.classList.add("failed");
    en.textContent = "Translation failed";
  } else if (line.done) {
    en.classList.add("failed");
    en.textContent = "(no translation)";
  } else {
    en.classList.add("waiting", "pending");
    en.textContent = waitingText(line);
  }
  if (line.done && !line.ok) {
    const b = document.createElement("button");
    b.className = "retry ghost";
    b.textContent = "↻ translate again";
    b.onclick = () => retranslate(meetingId || S.live.feedMeetingId, line.id, b);
    en.appendChild(b);
  }
  if (line.lat) {
    const meta = document.createElement("span");
    meta.className = "meta";
    const parts = [];
    if (line.lat.asr_s !== undefined) parts.push(`speech ${line.lat.asr_s}s`);
    if (line.lat.tr_first_s) parts.push(`EN starts ${line.lat.tr_first_s}s`);
    if (line.lat.tr_total_s) parts.push(`EN done ${line.lat.tr_total_s}s`);
    meta.textContent = "  · " + parts.join(" · ");
    $(".t", el).appendChild(document.createElement("br"));
    $(".t", el).appendChild(meta);
  }
}

async function retranslate(mid, lineId, btn) {
  if (!mid) return;
  if (btn) btn.disabled = true;
  try { await api(`/api/meetings/${encodeURIComponent(mid)}/lines/${lineId}/retranslate`, { body: {} }); }
  catch (e) { toast("error", e.message); if (btn) btn.disabled = false; }
}

function resetLive(meeting, lines) {
  S.live.meeting = meeting || null;
  S.live.feedMeetingId = meeting ? meeting.id : null;
  S.live.lines = new Map();
  S.live.els = new Map();
  const feed = $("#liveFeed");
  feed.innerHTML = "";
  const frag = document.createDocumentFragment();
  let prev = null;
  for (const ln of lines || []) {
    ln.cont = prev && prev.forced;
    S.live.lines.set(ln.id, ln);
    const el = lineEl(ln);
    S.live.els.set(ln.id, el);
    frag.appendChild(el);
    prev = ln;
  }
  feed.appendChild(frag);
  updateMeetingHeader();
  updateEmpty();
  scrollToEnd(true);
}

function waitingText(line) {
  const secs = line.t0 ? Math.floor((Date.now() - line.t0) / 1000) : 0;
  if (secs < 3) return "translating…";
  const llm = (S.status && S.status.llm) || {};
  let why = "";
  if (secs >= 12) {
    if (llm.warming || llm.loaded === false) why = secs >= 15 ? " - still loading the translation model (low on memory? close other apps)" : " - loading the translation model";
    else if (llm.queue > 1) why = ` - translator is ${llm.queue} lines behind`;
    else why = " - the translator is slow right now";
  }
  return `translating… ${secs} s${why}`;
}
// refresh the "translating… N s" counters once a second
setInterval(() => {
  for (const [id, ln] of S.live.lines) {
    if (ln.done || ln.en) continue;
    const el = S.live.els.get(id);
    const en = el && $(".en.waiting", el);
    if (en) en.textContent = waitingText(ln);
  }
}, 1000);

function addLiveLine(line) {
  if (!line.done && !line.en) line.t0 = Date.now();
  const last = [...S.live.lines.values()].pop();
  line.cont = !!(last && last.forced);
  S.live.lines.set(line.id, line);
  let el = S.live.els.get(line.id);
  if (el) { fillLine(el, line); return; }
  el = lineEl(line);
  el.classList.add("fresh");
  S.live.els.set(line.id, el);
  $("#liveFeed").appendChild(el);
  updateEmpty();
  scrollToEnd();
}

function onTr(ev) {
  const apply = (store, els, mid) => {
    const ln = store.get(ev.id);
    if (!ln) return;
    ln.en = ev.en;
    if (ev.final) { ln.done = true; ln.ok = ev.ok !== false; if (ev.lat) ln.lat = ev.lat; }
    const el = els.get(ev.id);
    if (el) fillLine(el, ln, mid);
  };
  if (ev.meeting === S.live.feedMeetingId) { apply(S.live.lines, S.live.els, ev.meeting); scrollToEnd(); }
  if (S.view && ev.meeting === S.view.meta.id) apply(S.view.lines, S.view.els, ev.meeting);
}

function setPartial(text) {
  const p = $("#partial");
  const listening = S.status && S.status.state === "listening";
  $(".de", p).textContent = text || "";
  p.hidden = !listening || !!S.view;
  if (text) scrollToEnd();
}

function updateEmpty() {
  const showing = S.view ? S.view.lines.size : S.live.lines.size;
  const listening = S.status && S.status.state !== "idle";
  $("#emptyState").hidden = !!(showing || listening || S.view);
}

// ------------------------------------------------------------------ scrolling
function nearBottom() {
  const sc = $("#scroller");
  return sc.scrollHeight - sc.scrollTop - sc.clientHeight < 140;
}
function scrollToEnd(force) {
  if (S.view && !force) return;
  if (!force && !S.follow) { $("#jumpLive").hidden = false; return; }
  requestAnimationFrame(() => {
    const sc = $("#scroller");
    sc.scrollTop = S.view ? 0 : sc.scrollHeight;
  });
}
$("#scroller").addEventListener("scroll", () => {
  if (S.view) return;
  S.follow = nearBottom();
  $("#jumpLive").hidden = S.follow;
}, { passive: true });
$("#jumpLive").onclick = () => { S.follow = true; $("#jumpLive").hidden = true; scrollToEnd(true); };

// ------------------------------------------------------------------ status / header
function applyStatus(st) {
  S.status = st;
  const btn = $("#startBtn");
  btn.disabled = false;
  btn.classList.remove("stop", "primary");
  if (st.state === "listening") { btn.textContent = "Stop"; btn.classList.add("stop"); }
  else if (st.state === "stopping") { btn.textContent = "Saving…"; btn.disabled = true; }
  else { btn.textContent = "Start"; btn.classList.add("primary"); }

  if (st.meeting && st.state !== "idle" && (!S.live.meeting || S.live.meeting.id === st.meeting.id)) S.live.meeting = st.meeting;
  updateMeetingHeader();

  // speech model pill
  const a = $("#asrPill");
  a.className = "pill " + (st.asr.status === "ready" ? "ok" : st.asr.status === "error" ? "err" : "warn");
  a.title = st.asr.detail || "";
  // translator pill
  const l = $("#llmPill");
  const llm = st.llm || {};
  let cls = "warn", tip = "Checking Ollama…";
  if (llm.checked && !llm.running) { cls = "err"; tip = "Ollama is not running - start the Ollama app (translation is off)."; }
  else if (llm.running && !llm.model_ready) { cls = "err"; tip = `Translation model ${llm.model} is not installed - click to download it in Settings.`; }
  else if (llm.running && llm.model_ready) {
    const speed = llm.tok_s ? ` · ${llm.tok_s} tok/s` : "";
    const fb = llm.fallback_active ? " (fast model - the main one was too slow on this Mac)" : "";
    if (llm.stuck) { cls = "err"; tip = `Translation is not answering right now - retrying automatically (${llm.model})${fb}`; }
    else if ((llm.warming || llm.busy) && !llm.loaded) { cls = "warn"; tip = `Loading translation model ${llm.model}…`; }
    else if (llm.speed === "slow") { cls = "warn"; tip = `${llm.model} is slow on this Mac right now${speed}${fb}`; }
    else if ((llm.queue || 0) > 2) { cls = "warn"; tip = `Translator is ${llm.queue} lines behind${speed}${fb}`; }
    else { cls = "ok"; tip = `Translating with ${llm.model}${speed}${fb}`; }
  }
  l.className = "pill " + cls;
  l.title = tip;
  $("#llmPill b").textContent = llm.fallback_active ? "Translator (fast)" : "Translator";

  // timer
  if (st.state === "listening" && st.meeting) {
    S.startedAt = new Date(st.meeting.started).getTime();
    if (!S.timer) S.timer = setInterval(tick, 1000);
    tick();
  } else if (st.state === "idle") {
    clearInterval(S.timer); S.timer = null;
    $("#timer").textContent = "00:00";
    $("#meter").classList.remove("clip");
    $("#meterFill").style.width = "0";
    $("#meterVoice").classList.remove("on");
  }
  updateBrowserAudioBar();
  $("#srcName").textContent = st.state === "listening" && st.source ? "Listening with: " + st.source : "";
  setPartial($("#partial .de").textContent);
  updateEmpty();
  const hint = [];
  if (st.asr.status === "loading") hint.push(st.asr.detail);
  if (st.asr.status === "error") hint.push(st.asr.detail);
  if (llm.checked && !llm.running) hint.push("Ollama is not running: German will still be shown and saved, but not translated.");
  $("#emptyHint").textContent = hint.join(" ");
}

function tick() {
  if (S.startedAt) $("#timer").textContent = fmtDur((Date.now() - S.startedAt) / 1000);
}

function updateMeetingHeader() {
  const m = S.live.meeting;
  const inp = $("#meetingName");
  if (document.activeElement !== inp) inp.value = m ? (m.name || "") : (inp.dataset.pending || "");
  inp.placeholder = m ? "Name this meeting…" : "Name the next meeting (optional)…";
  $("#meetingWhen").textContent = m ? fmtDate(m.started, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
}

function updateMeter(ev) {
  // -60 dBFS .. -6 dBFS -> 0..100 %
  const pct = Math.max(0, Math.min(100, ((ev.db + 60) / 54) * 100));
  $("#meterFill").style.width = pct + "%";
  $("#meter").classList.toggle("clip", ev.db > -6);
  $("#meterVoice").classList.toggle("on", !!ev.speech);
  $("#meter").title = `Level ${ev.db} dBFS · auto boost +${ev.boost || 0} dB`;
}

// ------------------------------------------------------------------ events
function onEvent(ev) {
  switch (ev.type) {
    case "snapshot":
      S.settings = ev.settings;
      applySettings();
      applyStatus(ev.status);
      resetLive(ev.status.meeting, ev.lines);
      setPartial(ev.partial && ev.partial.text);
      break;
    case "status": applyStatus(ev); break;
    case "session":
      resetLive(ev.meeting, ev.lines);
      if (S.view) closeView();
      break;
    case "session_end":
      if (ev.removed) toast("info", "Nothing was said - the empty meeting was not kept.");
      else if (ev.meeting) toast("ok", `Saved: ${ev.meeting.title} (${ev.meeting.line_count} lines)`);
      // the transcript stays on screen; the header is ready for the next meeting
      stopBrowserAudio();
      S.live.meeting = null;
      $("#meetingName").dataset.pending = "";
      updateMeetingHeader();
      setPartial("");
      break;
    case "meeting":
      if (S.live.meeting && ev.meeting.id === S.live.meeting.id) { S.live.meeting = ev.meeting; updateMeetingHeader(); }
      break;
    case "line": ev.line.lat = ev.lat; addLiveLine(ev.line); setPartial(""); break;
    case "tr": onTr(ev); break;
    case "partial": setPartial(ev.text); break;
    case "level": updateMeter(ev); break;
    case "notice": toast(ev.level, ev.text); break;
    case "browser_audio_needed": if (!BA.ctx) startBrowserAudio("mic", true); break;
    case "meetings_changed": if (!$("#meetingsDrawer").hidden) loadMeetings(); break;
    case "settings": S.settings = ev.settings; applySettings(); break;
    case "pull": onPull(ev); break;
    case "summary": onSummary(ev); break;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  S.ws = ws;
  let pinger = null;
  ws.onopen = () => {
    S.wsDelay = 500;
    pinger = setInterval(() => { try { ws.send("ping"); } catch (_) { /* closed */ } }, 20000);
    $$(".toast").forEach((t) => { if (t.dataset.text && t.dataset.text.startsWith("Connection lost")) t.remove(); });
  };
  ws.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.error(e); } };
  ws.onclose = () => {
    clearInterval(pinger);
    if (S.status) toast("warn", "Connection lost - reconnecting… (the meeting keeps being recorded)", 4000);
    setTimeout(connect, S.wsDelay);
    S.wsDelay = Math.min(S.wsDelay * 2, 5000);
  };
}

// ------------------------------------------------------------------ start / stop / rename
$("#startBtn").onclick = async () => {
  const st = S.status && S.status.state;
  const btn = $("#startBtn");
  btn.disabled = true;
  try {
    if (st === "listening") await api("/api/stop", { body: {} });
    else {
      const name = $("#meetingName").value.trim();
      await api("/api/start", { body: { name } });
      $("#meetingName").dataset.pending = "";
      S.follow = true;
    }
  } catch (e) { toast("error", e.message); btn.disabled = false; }
};

let renameTimer = null;
$("#meetingName").addEventListener("input", () => {
  const v = $("#meetingName").value;
  if (!S.live.meeting || (S.status && S.status.state === "idle")) { $("#meetingName").dataset.pending = v; return; }
  clearTimeout(renameTimer);
  renameTimer = setTimeout(() => saveName(v), 700);
});
$("#meetingName").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.target.blur(); } });
$("#meetingName").addEventListener("change", () => {
  if (S.live.meeting && S.status && S.status.state !== "idle") { clearTimeout(renameTimer); saveName($("#meetingName").value); }
});
async function saveName(v) {
  if (!S.live.meeting) return;
  try {
    const meta = await api(`/api/meetings/${encodeURIComponent(S.live.meeting.id)}/rename`, { body: { name: v } });
    if (meta) S.live.meeting = meta;
  } catch (e) { toast("error", e.message); }
}

// ------------------------------------------------------------------ text size / German toggle
function setScale(v) {
  v = Math.round(Math.max(0.7, Math.min(2.2, v)) * 100) / 100;
  document.documentElement.style.setProperty("--scale", v);
  saveSettings({ font_scale: v });
}
$("#smaller").onclick = () => setScale((S.settings ? S.settings.font_scale : 1) - 0.1);
$("#bigger").onclick = () => setScale((S.settings ? S.settings.font_scale : 1) + 0.1);
$("#toggleDe").onclick = () => saveSettings({ show_german: !(S.settings && S.settings.show_german) });

// ------------------------------------------------------------------ drawers
function openDrawer(id) {
  $$(".drawer").forEach((d) => (d.hidden = d.id !== id));
  $("#scrim").hidden = false;
}
function closeDrawers() { $$(".drawer").forEach((d) => (d.hidden = true)); $("#scrim").hidden = true; }
$("#scrim").onclick = closeDrawers;
$$("[data-close]").forEach((b) => (b.onclick = closeDrawers));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawers(); });
$("#meetingsBtn").onclick = () => { openDrawer("meetingsDrawer"); loadMeetings(); };
$("#settingsBtn").onclick = () => { openDrawer("settingsDrawer"); loadDevicesAndModels(); };
$("#asrPill").onclick = () => { if ($("#asrPill").classList.contains("err")) { openDrawer("settingsDrawer"); loadDevicesAndModels(); } };
$("#llmPill").onclick = () => { if ($("#llmPill").classList.contains("err")) { openDrawer("settingsDrawer"); loadDevicesAndModels(); } };

// ------------------------------------------------------------------ meetings list + viewer
async function loadMeetings() {
  try { S.meetings = await api("/api/meetings"); } catch (e) { toast("error", e.message); return; }
  renderMeetingList();
}
function renderMeetingList() {
  const q = $("#meetingSearch").value.trim().toLowerCase();
  const box = $("#meetingList");
  box.innerHTML = "";
  let day = null;
  const liveId = S.status && S.status.state !== "idle" && S.live.meeting ? S.live.meeting.id : null;
  const items = S.meetings.filter((m) => !q || (m.title || "").toLowerCase().includes(q) || (m.date || "").includes(q));
  if (!items.length) { box.innerHTML = '<p class="muted" style="padding:12px">No meetings yet.</p>'; return; }
  for (const m of items) {
    if (m.date !== day) {
      day = m.date;
      const h = document.createElement("div");
      h.className = "day";
      h.textContent = dayLabel(m.date);
      box.appendChild(h);
    }
    const b = document.createElement("button");
    b.className = "mitem" + (m.id === liveId ? " live" : "");
    const start = fmtDate(m.started, { hour: "2-digit", minute: "2-digit" });
    const end = m.ended ? "–" + fmtDate(m.ended, { hour: "2-digit", minute: "2-digit" }) : "";
    b.innerHTML = '<span class="mt"></span><span class="ms"></span>';
    $(".mt", b).textContent = m.title;
    $(".ms", b).textContent = `${start}${end} · ${m.line_count} lines · ${fmtDur(m.duration_s)}${m.has_summary ? " · notes" : ""}`;
    b.onclick = () => { closeDrawers(); if (m.id === liveId) closeView(); else openView(m.id); };
    box.appendChild(b);
  }
}
$("#meetingSearch").addEventListener("input", renderMeetingList);

async function openView(id) {
  let data;
  try { data = await api(`/api/meetings/${encodeURIComponent(id)}`); } catch (e) { toast("error", e.message); return; }
  const lines = new Map(), els = new Map();
  const feed = $("#viewFeed");
  feed.innerHTML = "";
  let prev = null;
  for (const ln of data.lines) {
    ln.cont = prev && prev.forced;
    lines.set(ln.id, ln);
    const el = lineEl(ln);
    fillLine(el, ln, data.meta.id);
    els.set(ln.id, el);
    feed.appendChild(el);
    prev = ln;
  }
  S.view = { meta: data.meta, lines, els };
  $("#vbTitle").textContent = data.meta.title;
  $("#vbWhen").textContent = `${fmtDate(data.meta.started, { weekday: "long", day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit" })} · ${data.meta.line_count} lines`;
  const base = `/api/meetings/${encodeURIComponent(id)}/download`;
  $("#vbDownloadMd").href = base + "?fmt=md";
  $("#vbDownloadTxt").href = base + "?fmt=txt";
  $("#vbContinue").disabled = S.status && S.status.state !== "idle";
  $("#viewBanner").hidden = false;
  $("#liveFeed").hidden = true;
  feed.hidden = false;
  showSummary(data.summary, false);
  $("#jumpLive").hidden = true;
  setPartial("");
  updateEmpty();
  $("#scroller").scrollTop = 0;
}
function closeView() {
  S.view = null;
  $("#viewBanner").hidden = true;
  $("#viewFeed").hidden = true;
  $("#viewFeed").innerHTML = "";
  $("#liveFeed").hidden = false;
  $("#summaryBox").hidden = true;
  S.follow = true;
  updateEmpty();
  setPartial($("#partial .de").textContent);
  scrollToEnd(true);
}
$("#vbBack").onclick = closeView;
$("#vbDelete").onclick = async () => {
  if (!S.view) return;
  if (!confirm(`Delete "${S.view.meta.title}" permanently?\n\nThe transcript files will be removed from disk.`)) return;
  try {
    await api(`/api/meetings/${encodeURIComponent(S.view.meta.id)}`, { method: "DELETE" });
    toast("ok", "Meeting deleted.");
    closeView();
  } catch (e) { toast("error", e.message); }
};
$("#vbFolder").onclick = async () => {
  if (!S.view) return;
  try { await api(`/api/meetings/${encodeURIComponent(S.view.meta.id)}/open-folder`, { body: {} }); }
  catch (e) { toast("warn", `${e.message}. Folder: ${S.view.meta.folder}`); }
};
$("#vbContinue").onclick = async () => {
  if (!S.view) return;
  try { await api("/api/start", { body: { continue_id: S.view.meta.id } }); S.follow = true; }
  catch (e) { toast("error", e.message); }
};
$("#vbSummary").onclick = async () => {
  if (!S.view) return;
  showSummary("Writing meeting notes… (this can take a minute for long meetings)", true);
  try { await api(`/api/meetings/${encodeURIComponent(S.view.meta.id)}/summary`, { body: {} }); }
  catch (e) { showSummary(null); toast("error", e.message); }
};
function showSummary(text, working) {
  const box = $("#summaryBox");
  if (!text) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  box.innerHTML = '<h3>Meeting notes</h3><div class="md"></div>';
  $(".md", box).textContent = text.replace(/^# .*\n+/, "");
  if (working) box.classList.add("working"); else box.classList.remove("working");
}
function onSummary(ev) {
  if (!S.view || S.view.meta.id !== ev.id) { if (ev.done && !ev.error) toast("ok", "Meeting notes are ready."); return; }
  if (ev.error) { showSummary(null); toast("error", "Notes failed: " + ev.error); return; }
  showSummary(ev.text, !ev.done);
}

// ------------------------------------------------------------------ settings
function applySettings() {
  const s = S.settings;
  if (!s) return;
  document.documentElement.style.setProperty("--scale", s.font_scale);
  document.body.classList.toggle("hide-de", !s.show_german);
  $("#toggleDe").setAttribute("aria-pressed", String(!!s.show_german));
  if (s.theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", s.theme);
  const f = $("#settingsForm");
  for (const el of f.elements) {
    if (!el.name || !(el.name in s)) continue;
    if (document.activeElement === el) continue;
    if (el.type === "checkbox") el.checked = !!s[el.name];
    else if (el.tagName === "SELECT") {
      if (![...el.options].some((o) => o.value === String(s[el.name]))) {
        const o = document.createElement("option");
        o.value = s[el.name]; o.textContent = s[el.name];
        el.appendChild(o);
      }
      el.value = String(s[el.name]);
    } else el.value = s[el.name];
  }
  $("#pauseOut").textContent = (s.pause_ms / 1000).toFixed(2) + " s";
  $("#fontOut").textContent = Math.round(s.font_scale * 100) + "%";
  $$("[data-for-source]").forEach((el) => (el.hidden = el.dataset.forSource !== s.input_source));
  updatePullBox();
  updateBrowserAudioBar();
}

function updateBrowserAudioBar() {
  const st = S.status;
  const show = !!(st && st.state === "listening" && (st.source_kind === "browser" || (S.settings && S.settings.input_source === "browser")));
  $("#browserAudio").hidden = !show;
  if (BA.ctx && st && st.state === "listening" && st.source_kind && st.source_kind !== "browser") stopBrowserAudio();
  if (!show) return;
  const fallback = S.settings && S.settings.input_source === "mic";
  $("#baText").innerHTML = fallback
    ? "macOS gives no sound from the Mac microphone - using <b>this browser's microphone</b>."
    : "Audio input is set to <b>this browser</b>.";
}

let saveTimer = null, pendingChanges = {};
function saveSettings(changes, delay = 0) {
  Object.assign(pendingChanges, changes);
  if (S.settings) Object.assign(S.settings, changes);
  applySettings();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const body = pendingChanges;
    pendingChanges = {};
    try {
      S.settings = await api("/api/settings", { body });
      applySettings();
      const m = $("#savedMark");
      m.hidden = false;
      setTimeout(() => (m.hidden = true), 1200);
    } catch (e) { toast("error", "Settings not saved: " + e.message); }
  }, delay);
}
$("#settingsForm").addEventListener("input", (e) => {
  const el = e.target;
  if (!el.name) return;
  let v = el.type === "checkbox" ? el.checked : el.value;
  if (el.type === "range" || el.type === "number") v = Number(v);
  const textual = el.tagName === "TEXTAREA" || el.type === "number";
  saveSettings({ [el.name]: v }, textual ? 700 : el.type === "range" ? 250 : 0);
});

async function loadDevicesAndModels() {
  try {
    const [devs, models, info] = await Promise.all([api("/api/devices"), api("/api/models"), api("/api/info")]);
    const ds = $("#deviceSelect");
    ds.innerHTML = '<option value="">System default</option>';
    for (const d of devs) {
      const o = document.createElement("option");
      o.value = d.name; o.textContent = d.name + (d.default ? "  (default)" : "");
      ds.appendChild(o);
    }
    const as = $("#asrSelect");
    as.innerHTML = "";
    for (const [k, label] of Object.entries(models.asr)) {
      const o = document.createElement("option");
      o.value = k; o.textContent = label;
      as.appendChild(o);
    }
    S.models = models;
    const ls = $("#llmSelect");
    ls.innerHTML = "";
    const seen = new Set();
    const installed = new Set(models.llm_installed);
    const add = (name, label) => {
      if (seen.has(name)) return;
      seen.add(name);
      const o = document.createElement("option");
      o.value = name;
      o.textContent = label + (installed.has(name) ? "" : "  - not installed");
      ls.appendChild(o);
    };
    for (const r of models.llm_recommended) add(r.name, r.label);
    for (const m of models.llm_installed) add(m, m);
    const fs = $("#llmFallbackSelect");
    fs.innerHTML = ls.innerHTML;
    $("#infoLine").textContent = `Meetings are saved in ${info.meetings_dir}`;
    applySettings();
  } catch (e) { toast("error", e.message); }
}

function updatePullBox() {
  const s = S.settings, m = S.models;
  if (!s || !m) return;
  const installed = new Set(m.llm_installed);
  const want = s.llm_model.includes(":") ? s.llm_model : s.llm_model + ":latest";
  const missing = m.ollama_running && !installed.has(s.llm_model) && !installed.has(want);
  const box = $("#pullBox");
  if (!box.dataset.pulling) box.hidden = !missing;
  $("#pullText").textContent = missing ? `${s.llm_model} is not installed yet.` : "";
  if (!m.ollama_running) { box.hidden = false; $("#pullBtn").hidden = true; $("#pullText").textContent = "Ollama is not running - start the Ollama app."; }
  else $("#pullBtn").hidden = false;
}
$("#pullBtn").onclick = async () => {
  const model = S.settings.llm_model;
  $("#pullBox").dataset.pulling = "1";
  $("#pullBtn").disabled = true;
  $("#pullProgress").hidden = false;
  try { await api("/api/models/pull", { body: { model } }); } catch (e) { toast("error", e.message); }
};
function onPull(ev) {
  const box = $("#pullBox");
  box.hidden = false;
  if (ev.total) { $("#pullProgress").value = Math.round((ev.completed || 0) * 100 / ev.total); }
  $("#pullText").textContent = `${ev.model}: ${ev.status || ""}${ev.total ? " " + $("#pullProgress").value + "%" : ""}`;
  if (ev.done) {
    delete box.dataset.pulling;
    $("#pullBtn").disabled = false;
    $("#pullProgress").hidden = true;
    if (ev.status === "success") { toast("ok", `${ev.model} is ready.`); loadDevicesAndModels(); }
    else toast("error", `Download failed: ${ev.error || ev.status}`);
  }
}

// ------------------------------------------------------------------ browser audio (remote / host-name use)
const BA = { ctx: null, ws: null, stream: null, node: null, gen: 0 };

// resume() can stay pending forever when the browser wants a click first: never await it unbounded
function waitRunning(ctx, ms = 2000) {
  if (ctx.state === "running") return Promise.resolve(true);
  return new Promise((resolve) => {
    let t;
    const on = () => { if (ctx.state === "running") fin(true); };
    const fin = (ok) => { clearTimeout(t); ctx.removeEventListener("statechange", on); resolve(ok); };
    ctx.addEventListener("statechange", on);
    ctx.resume().then(on, () => {});
    t = setTimeout(() => fin(ctx.state === "running"), ms);
  });
}

async function startBrowserAudio(kind, automatic) {
  // an automatic start only in the tab the user is looking at (several tabs may be open);
  // a hidden tab starts as soon as the user looks at it again
  if (automatic && document.visibilityState !== "visible") {
    const later = () => {
      if (document.visibilityState !== "visible") return;
      document.removeEventListener("visibilitychange", later);
      const st = S.status;
      if (st && st.state === "listening" && st.source_kind === "browser" && !BA.ctx) startBrowserAudio(kind, true);
    };
    document.addEventListener("visibilitychange", later);
    return;
  }
  stopBrowserAudio();
  const gen = ++BA.gen;
  const stale = () => gen !== BA.gen || !(S.status && S.status.state === "listening");
  let stream = null, ctx = null, ws = null;
  const cleanup = () => {
    try { stream && stream.getTracks().forEach((t) => t.stop()); } catch (_) { /* */ }
    try { ws && ws.close(); } catch (_) { /* */ }
    try { ctx && ctx.close(); } catch (_) { /* */ }
  };
  try {
    if (kind === "tab") {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      stream.getVideoTracks().forEach((t) => t.stop());
      if (!stream.getAudioTracks().length) throw new Error("No audio shared - tick 'Share tab audio' / 'Share system audio'.");
    } else {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 1 } } });
    }
    if (stale()) { cleanup(); return; }
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/audio`);
    ws.binaryType = "arraybuffer";
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("audio connection failed")); });
    if (stale()) { cleanup(); return; }
    ws.send(JSON.stringify({ rate: ctx.sampleRate, format: "i16" }));
    const src = ctx.createMediaStreamSource(stream);
    const send = (f32) => {
      if (ws.readyState !== 1 || ws.bufferedAmount > 2e6) return;
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) { const v = Math.max(-1, Math.min(1, f32[i])); i16[i] = v < 0 ? v * 32768 : v * 32767; }
      ws.send(i16.buffer);
    };
    let node;
    if (ctx.audioWorklet) {
      const code = `class P extends AudioWorkletProcessor{constructor(){super();this.b=new Float32Array(4096);this.n=0}
        process(i){const c=i[0]&&i[0][0];if(c){if(this.n+c.length>this.b.length){this.flush()}this.b.set(c,this.n);this.n+=c.length;
        if(this.n>=2048){this.flush()}}return true}
        flush(){if(!this.n)return;const o=this.b.slice(0,this.n);this.port.postMessage(o,[o.buffer]);this.n=0}}
        registerProcessor('lt-pcm',P);`;
      const url = URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      if (stale()) { cleanup(); return; }
      node = new AudioWorkletNode(ctx, "lt-pcm", { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1, channelCountMode: "explicit", channelInterpretation: "speakers" });
      node.port.onmessage = (e) => send(e.data);
      src.connect(node);
    } else {
      node = ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) => send(e.inputBuffer.getChannelData(0));
      src.connect(node);
      node.connect(ctx.destination);
    }
    Object.assign(BA, { ctx, ws, stream, node });
    const label = (stream.getAudioTracks()[0] && stream.getAudioTracks()[0].label) || "microphone";
    const sending = kind === "tab" ? "● sending shared audio" : "● sending: " + label;
    $("#baState").textContent = sending;
    stream.getAudioTracks()[0].onended = stopBrowserAudio;
    ws.onclose = () => { if (BA.ws === ws) stopBrowserAudio(); };
    const running = await waitRunning(ctx);
    if (gen !== BA.gen || BA.ctx !== ctx) return;   // superseded meanwhile
    if (!running) {
      // the browser wants a click before audio may run
      $("#baState").textContent = "▶ click anywhere on this page to start the microphone";
      toast("warn", "Click anywhere on this page to start the browser microphone.");
      const go = () => {
        ctx.resume().catch(() => {});   // first statement inside the click: counts as a user gesture
        if (BA.ctx !== ctx) { document.removeEventListener("click", go, true); return; }
        waitRunning(ctx, 1500).then((ok) => {
          if (ok && BA.ctx === ctx) { $("#baState").textContent = sending; document.removeEventListener("click", go, true); }
        });
      };
      document.addEventListener("click", go, true);
    }
  } catch (e) {
    cleanup();
    if (gen !== BA.gen) return;
    let msg = e && e.name === "NotAllowedError"
      ? "The browser was not allowed to use the microphone. Click the microphone/lock icon in the address bar → Allow, "
        + "and check System Settings → Privacy & Security → Microphone → your browser."
      : (e.message || String(e));
    if (!window.isSecureContext) msg += " (needs https:// or localhost)";
    toast("error", "Browser microphone: " + msg);
    stopBrowserAudio();
  }
}
function stopBrowserAudio() {
  BA.gen++;
  try { BA.node && BA.node.disconnect(); } catch (_) { /* */ }
  try { BA.stream && BA.stream.getTracks().forEach((t) => t.stop()); } catch (_) { /* */ }
  try { BA.ws && BA.ws.close(); } catch (_) { /* */ }
  try { BA.ctx && BA.ctx.close(); } catch (_) { /* */ }
  Object.assign(BA, { ctx: null, ws: null, stream: null, node: null });
  $("#baState").textContent = "";
}
$("#baMic").onclick = () => startBrowserAudio("mic");

// ------------------------------------------------------------------ microphone test
$("#micTestBtn").onclick = async () => {
  const btn = $("#micTestBtn"), out = $("#micTestOut");
  btn.disabled = true;
  btn.textContent = "Testing… (play some sound)";
  out.innerHTML = "";
  try {
    const r = await api("/api/mictest", { body: {} });
    const perm = { authorized: "allowed", denied: "BLOCKED - System Settings → Privacy & Security → Microphone → Terminal", restricted: "blocked by a policy", not_determined: "not asked yet", unknown: "" }[r.permission];
    if (perm) { const p = document.createElement("p"); p.className = "small"; p.textContent = "macOS microphone permission: " + perm; out.appendChild(p); }
    for (const d of r.devices) {
      const row = document.createElement("div");
      row.className = "mic-row" + (d.ok ? "" : " bad");
      const pct = d.peak_db === null ? 0 : Math.max(3, Math.min(100, (d.peak_db + 70) / 70 * 100));
      row.innerHTML = '<span class="name"></span><span class="lvl"><i></i></span><span class="act"></span><span class="why"></span>';
      $(".name", row).textContent = d.name + (d.default ? " (system default)" : "");
      $(".lvl i", row).style.width = pct + "%";
      $(".why", row).textContent = d.error ? "error: " + d.error : d.ok ? `hears sound (peak ${d.peak_db} dB)` : "silent - blocked or a virtual device";
      if (d.ok) {
        const use = document.createElement("button");
        use.type = "button"; use.className = "ghost"; use.textContent = "Use";
        use.onclick = () => { saveSettings({ input_device: d.name }); $("#deviceSelect").value = d.name; };
        $(".act", row).appendChild(use);
      }
      out.appendChild(row);
    }
  } catch (e) { toast("error", e.message); }
  btn.disabled = false;
  btn.textContent = "Test microphones";
};
$("#baTab").onclick = () => startBrowserAudio("tab");

// ------------------------------------------------------------------ boot
api("/api/info").then((i) => { $("#dataDir").textContent = i.meetings_dir; }).catch(() => {});
connect();
