const $ = (s, el = document) => el.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const api = async (url, opts = {}) => {
  const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
};
const STAGES = ["research", "moments", "story", "voice", "edit", "timeline", "download", "render", "publish"];
const STAGE_NAMES = { research: "Search", moments: "Screen & understand", story: "Plan scenes", voice: "Hindi voice",
  edit: "Cut scenes", timeline: "Timeline", download: "Download", render: "Render", publish: "Upload kit" };
const MODE_NAMES = { narration: "Narrator", original: "Clip audio", music: "Music only" };
const fmt = (s) => { s = Math.round(s); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };

let selected = localStorage.getItem("sm-selected") || null;
let lastDetailKey = "";

// ---------------------------------------------------------------- form
document.querySelectorAll(".chips").forEach((group) => {
  group.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    group.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  });
});
const minutes = $("input[name=minutes]");
minutes.addEventListener("input", () => ($("#minutes-out").textContent = minutes.value));

$("#new-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const chip = (n) => $(`.chips[data-name=${n}] .on`)?.dataset.value;
  const body = {
    topic: f.topic.value, description: f.description.value, minutes: +f.minutes.value,
    theme: chip("theme"), narration: chip("narration"), voice: chip("voice"),
    style: f.style.value, resolution: f.resolution.value,
    review: f.review.checked, demo: f.demo.checked,
  };
  if (body.demo) body.minutes = 1.5;
  $("#form-error").textContent = "";
  const btn = $("button[type=submit]", f);
  btn.disabled = true;
  btn.textContent = "Starting…";
  try {
    const { id } = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    btn.textContent = "✓ Started – progress is on the right";
    select(id);
    refreshJobs();
  } catch (err) {
    $("#form-error").textContent = err.message;
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = "🎬 Make my video"; }, 4000);
});

// ---------------------------------------------------------------- status + settings
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const tracks = Object.values(s.music).reduce((a, b) => a + b, 0);
    const pills = [
      [s.ffmpeg ? (s.ffmpeg_text ? "ok" : "warn") : "bad",
        s.ffmpeg ? (s.ffmpeg_text ? "ffmpeg" : "ffmpeg: no Hindi text (brew install ffmpeg-full)") : "ffmpeg missing"],
      [s.yt_dlp ? "ok" : "bad", s.yt_dlp ? `yt-dlp ${s.yt_dlp}` : "yt-dlp missing"],
      [s.js_runtime ? "ok" : "warn", s.js_runtime ? "JS runtime" : "install deno for YouTube"],
      [s.llm_ready ? "ok" : (s.ollama ? "warn" : "bad"),
        s.llm_ready ? "AI ready" : (s.ollama ? "pick a model in Settings" : "Ollama off: basic mode")],
      [tracks ? "ok" : "warn", tracks ? `${tracks} music tracks` : "no music tracks yet"],
    ];
    $("#pills").innerHTML = pills.map(([c, t]) => `<span class="pill ${c}">${esc(t)}</span>`).join("");
    window.__models = s.models;
  } catch { $("#pills").innerHTML = `<span class="pill bad">server offline</span>`; }
}

$("#open-settings").addEventListener("click", async () => {
  const s = await api("/api/settings");
  const f = $("#settings-form");
  const models = window.__models || [];
  $("#model-select").innerHTML = [...new Set([s.llm_model, ...models])]
    .map((m) => `<option ${m === s.llm_model ? "selected" : ""}>${esc(m)}</option>`).join("");
  for (const el of f.elements) {
    if (!el.name || !(el.name in s)) continue;
    if (el.type === "checkbox") el.checked = !!s[el.name];
    else el.value = Array.isArray(s[el.name]) ? s[el.name].join(", ") : s[el.name];
  }
  $("#settings").showModal();
});
$("#save-settings").addEventListener("click", async (e) => {
  e.preventDefault();
  const f = $("#settings-form");
  const body = {};
  for (const el of f.elements) {
    if (!el.name) continue;
    body[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  await api("/api/settings", { method: "POST", body: JSON.stringify(body) });
  $("#settings").close();
  refreshStatus();
});

// ---------------------------------------------------------------- styles
async function refreshStyles() {
  const { styles, tasks } = await api("/api/styles");
  const sel = $("#style-select");
  const cur = sel.value || "cinematic";
  sel.innerHTML = styles.map((s) =>
    `<option value="${esc(s.name)}" ${s.name === cur ? "selected" : ""}>${esc(s.name)}${s.source === "built-in" ? " (built-in)" : ""}</option>`).join("");
  const items = styles.filter((s) => s.source !== "built-in").map((s) => {
    const st = s.stats || {};
    return `<div class="style-item"><div><b>${esc(s.name)}</b><br><span class="muted">${esc(s.source).slice(0, 70)}<br>
      ${st.shots || "?"} shots · avg ${st.avg_shot || "?"}s · dialogue ${Math.round((s.dialogue_ratio || 0) * 100)}% ·
      climax shots ${s.acts.climax.shot}s</span></div>
      <button class="ghost small" data-del-style="${esc(s.name)}">Remove</button></div>`;
  });
  const running = Object.entries(tasks).filter(([, t]) => t.status !== "done").map(([n, t]) =>
    `<div class="style-item"><span>${esc(n)}: ${t.status === "running" ? "downloading & analysing…" : "failed – " + esc(t.error)}</span></div>`);
  $("#styles").innerHTML = running.join("") + items.join("");
}
$("#style-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  try {
    await api("/api/styles", { method: "POST", body: JSON.stringify({ url: f.url.value, name: f.name.value }) });
    f.reset();
  } catch (err) { alert(err.message); }
  refreshStyles();
});
$("#styles").addEventListener("click", async (e) => {
  const name = e.target.dataset.delStyle;
  if (name && confirm(`Remove style "${name}"?`)) {
    await api(`/api/styles/${encodeURIComponent(name)}`, { method: "DELETE" });
    refreshStyles();
  }
});

// ---------------------------------------------------------------- jobs
async function refreshJobs() {
  const jobs = await api("/api/jobs");
  if (!jobs.length) return;
  $("#jobs").innerHTML = jobs.map((j) => `
    <div class="job ${j.id === selected ? "sel" : ""}" data-id="${esc(j.id)}">
      <div class="t">${esc(j.topic)}</div>
      <span class="badge ${esc(j.status)}">${esc(j.status === "awaiting_review" ? "review script" : j.status)}</span>
      <div class="bar"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div>
    </div>`).join("");
  if (!selected && jobs[0]) select(jobs[0].id);
}
$("#jobs").addEventListener("click", (e) => {
  const el = e.target.closest(".job");
  if (el) select(el.dataset.id);
});

function select(id) {
  selected = id;
  lastDetailKey = "";
  try { localStorage.setItem("sm-selected", id); } catch {}
  document.querySelectorAll(".job").forEach((j) => j.classList.toggle("sel", j.dataset.id === id));
  refreshDetail();
}

async function refreshDetail() {
  if (!selected) return;
  let d;
  try { d = await api(`/api/jobs/${encodeURIComponent(selected)}`); }
  catch { selected = null; $("#detail").hidden = true; return; }
  const st = d.state;
  const key = `${st.status}|${st.stage}|${st.progress}|${(st.log || []).length}`;
  if (key === lastDetailKey) return;
  // Don't rebuild while the user is editing narration.
  if (st.status === "awaiting_review" && lastDetailKey.startsWith("awaiting_review")) return;
  lastDetailKey = key;
  const el = $("#detail");
  el.hidden = false;
  const idx = STAGES.indexOf(st.stage);
  const steps = STAGES.map((s, i) => {
    const cls = st.status === "done" || i < idx ? "done" : (i === idx && st.status === "running" ? "now" : "");
    return `<span class="step ${cls}">${STAGE_NAMES[s]}</span>`;
  }).join("");
  const base = `/jobs/${encodeURIComponent(st.id)}`;
  let html = `<div class="copyrow"><h2>${esc(st.request.topic)}</h2><div>`;
  if (["running", "queued"].includes(st.status)) html += `<button class="ghost small" data-act="cancel">Cancel</button>`;
  if (["failed", "cancelled"].includes(st.status)) html += `<button class="secondary small" data-act="retry">Resume</button> `;
  if (st.status !== "running") html += ` <button class="ghost small" data-act="delete">Delete</button>`;
  html += `</div></div><div class="steps">${steps}</div>`;
  if (["running", "queued"].includes(st.status)) {
    const last = (st.log || []).filter((l) => !l.startsWith("Traceback")).slice(-1)[0] || "Waiting to start…";
    const mins = Math.max(0, Math.round((Date.now() / 1000 - (st.started || st.created)) / 60));
    html += `<div class="now"><span class="spinner"></span><div><b>Now:</b> ${esc(last.replace(/^\d\d:\d\d:\d\d\s+/, ""))}
      <div class="muted">${Math.round((st.progress || 0) * 100)}% · running ${mins} min · you can close this tab, it keeps going</div></div></div>`;
  }
  if (st.status === "failed") html += `<div class="banner bad"><b>Stopped:</b> ${esc(st.error)}<br>Fix the cause and press <b>Resume</b> – finished steps are kept.</div>`;
  if (st.status === "awaiting_review") html += `<div class="banner">Your story is ready. Read the Hindi narration below, change any line you like, then press <b>Render video</b>.</div>`;

  if (st.status === "done" && st.outputs) {
    html += `<video controls preload="metadata" src="${base}/final.mp4" poster="${base}/thumbnail.jpg"></video>
      <div class="downloads">
        <a class="btn small" href="${base}/final.mp4?download=1">⬇ Video (${fmt(st.outputs.duration)})</a>
        <a class="btn small" href="${base}/thumbnail.jpg?download=1">⬇ Thumbnail</a>
        <a class="btn small" href="${base}/narration_hi.srt?download=1">⬇ Hindi subtitles (.srt)</a>
        <a class="btn small" href="${base}/youtube.txt?download=1">⬇ Upload kit</a>
      </div>`;
  }
  if (d.youtube) {
    html += `<details open class="kit"><summary>YouTube upload kit</summary>
      ${["title", "description"].map((k) => `<div class="copyrow"><b>${k}</b><button class="ghost small" data-copy="${k}">Copy</button></div>
      <textarea readonly rows="${k === "title" ? 2 : 8}" id="kit-${k}">${esc(d.youtube[k])}</textarea>`).join("")}
      <div class="copyrow"><b>tags</b><button class="ghost small" data-copy="tags">Copy</button></div>
      <textarea readonly rows="2" id="kit-tags">${esc(d.youtube.tags.join(", "))}</textarea></details>`;
  }
  if (d.story) html += storyboard(d.story, st.status === "awaiting_review");
  if (st.status === "awaiting_review") html += `<button class="primary" data-act="approve">🎬 Render video</button>`;
  if (d.research) {
    html += `<details><summary>Research: looked at ${d.research.total} videos, studied ${d.research.shortlist.length}</summary>
      <p class="muted">Searches: ${d.research.queries.map(esc).join(" · ")}</p>
      <ol class="muted">${d.research.shortlist.map((v) => `<li><a href="https://www.youtube.com/watch?v=${esc(v.id)}" target="_blank" rel="noopener">${esc(v.title)}</a> – ${esc(v.channel)}</li>`).join("")}</ol></details>`;
  }
  if (d.rejected && d.rejected.length) {
    html += `<details><summary>Rejected ${d.rejected.length} videos (off-topic or language)</summary>
      <ul class="muted">${d.rejected.map((v) => `<li><a href="https://www.youtube.com/watch?v=${esc(v.id)}" target="_blank" rel="noopener">${esc(v.title)}</a> – ${esc(v.reason)}</li>`).join("")}</ul></details>`;
  }
  html += `<details ${st.status === "failed" ? "open" : ""}><summary>Log</summary><pre class="log">${esc((st.log || []).join("\n"))}</pre></details>`;
  el.innerHTML = html;
  const log = $("pre.log", el);
  if (log) log.scrollTop = log.scrollHeight;
}

const KIND_NAMES = { hook: "Hook", dialogue: "Clip audio", narration: "Narrator", text: "Text on screen",
  montage: "Montage", voiceover: "Hindi voice-over" };

function storyboard(story, editable) {
  let html = `<h3>Storyboard – “${esc(story.title_hi)}”${story.source === "template" ? " (built-in, no AI)" : ""}</h3>`;
  const r = story.report;
  if (r && (r.score || (r.issues || []).length)) {
    html += `<div class="banner ${r.score && r.score < 6 ? "bad" : ""}"><b>Editor's self-check: ${r.score || "?"}/10</b>
      ${(r.issues || []).length ? `<ul>${r.issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : ""}</div>`;
  }
  for (const w of story.warnings || []) html += `<div class="banner bad">${esc(w)}</div>`;
  if (editable) html += `<p class="muted">Untick <b>Keep</b> to drop a scene. Text in boxes can be edited.</p>`;
  for (const act of story.acts) {
    const secs = act.beats.reduce((a, b) => a + (b.seconds || 0), 0);
    html += `<div class="act"><h4>${esc(act.title_hi)} <small>${esc(act.key)} · ${fmt(secs)}</small></h4>`;
    for (const b of act.beats) {
      const kind = b.kind || (b.teaser ? "hook" : b.audio);
      const clips = (b.clips || []).map((c) =>
        `<a href="https://www.youtube.com/watch?v=${esc(c.video_id)}&t=${Math.floor(c.start)}s" target="_blank" rel="noopener" title="${esc(c.video_title)}">
          <img loading="lazy" src="https://i.ytimg.com/vi/${esc(c.video_id)}/mqdefault.jpg" alt="" onerror="this.style.visibility='hidden'"><span>${fmt(c.end - c.start)}</span></a>`).join("");
      let body = "";
      if (kind === "narration" || kind === "text" || kind === "voiceover") {
        body = editable ? `<textarea rows="2" data-scene="${esc(b.scene_id)}">${esc(b.narration)}</textarea>`
                        : `<div class="narr">${kind === "text" ? "▣ " : ""}“${esc(b.narration)}”</div>`;
      }
      if (b.said) body += `<div class="said"><b>${esc(b.source)}</b><br>“${esc(b.said.slice(0, 260))}${b.said.length > 260 ? "…" : ""}”</div>`;
      const keep = editable && b.scene_id ? `<label class="keep"><input type="checkbox" checked data-keep="${esc(b.scene_id)}"> Keep</label>` : "";
      html += `<div class="beat"><div class="mode ${esc(b.audio)}">${KIND_NAMES[kind] || esc(kind)}<br><small>${fmt(b.seconds || 0)}</small>${keep}</div>
        <div>${body}${b.idea ? `<div class="idea">↳ ${esc(b.idea)}</div>` : ""}<div class="thumbs">${clips}</div></div></div>`;
    }
    html += `</div>`;
  }
  return html;
}

$("#detail").addEventListener("click", async (e) => {
  const act = e.target.dataset.act;
  const copy = e.target.dataset.copy;
  if (copy) {
    const t = $(`#kit-${copy}`);
    navigator.clipboard?.writeText(t.value);
    e.target.textContent = "Copied";
    return;
  }
  if (!act || !selected) return;
  const url = `/api/jobs/${encodeURIComponent(selected)}`;
  if (act === "cancel") await api(`${url}/cancel`, { method: "POST" });
  if (act === "retry") await api(`${url}/retry`, { method: "POST" });
  if (act === "delete") {
    if (!confirm("Delete this video and all its files?")) return;
    await api(url, { method: "DELETE" });
    selected = null;
    $("#detail").hidden = true;
    $("#jobs").innerHTML = "";
  }
  if (act === "approve") {
    const edits = {};
    document.querySelectorAll("#detail textarea[data-scene]").forEach((t) => (edits[t.dataset.scene] = t.value));
    const remove = [...document.querySelectorAll("#detail input[data-keep]")].filter((c) => !c.checked)
      .map((c) => c.dataset.keep);
    await api(`${url}/approve`, { method: "POST", body: JSON.stringify({ edits, remove }) });
  }
  lastDetailKey = "";
  refreshJobs();
  refreshDetail();
});

// ---------------------------------------------------------------- loop
refreshStatus();
refreshStyles();
refreshJobs().then(refreshDetail);
setInterval(() => { refreshJobs(); refreshDetail(); }, 2500);
setInterval(refreshStatus, 20000);
setInterval(refreshStyles, 6000);
