"use strict";

const STAGES = ["queued", "downloading", "separating", "done"];
// The stages that report a fraction to fill a bar with.
const MEASURED = ["downloading", "separating"];

const el = (id) => document.getElementById(id);
const jobsEl = el("jobs");

/** Live EventSource per job id, so a re-render does not open a second one. */
const streams = new Map();

// ------------------------------------------------------------------ health

async function loadHealth() {
  let health;
  try {
    health = await (await fetch("/api/health")).json();
  } catch {
    setPill("offline", "error");
    return null;
  }

  const worker = health.worker;
  if (worker.running) {
    const loaded = worker.models.filter((m) => m.status === "loaded");
    const device = loaded.length ? loaded[0].device : "?";
    setPill(`${loaded.length} model${loaded.length === 1 ? "" : "s"} on ${device}`, "ok");
  } else {
    setPill("worker down", "error");
    const w = el("worker-warning");
    w.textContent = "The worker is not running. Restart the server.";
    w.hidden = false;
  }

  const authed = health.tidal.authenticated;
  el("auth-warning").hidden = authed;
  el("login-command").textContent = health.tidal.login_command;
  // Nothing can be submitted without a Tidal session or a worker.
  el("submit").disabled = !authed || !worker.running;

  return health;
}

function setPill(text, kind) {
  const pill = el("health");
  pill.textContent = text;
  pill.className = `pill pill--${kind}`;
}

async function loadModels(health) {
  const select = el("model");
  const { models } = await (await fetch("/api/models")).json();
  const usable = models.filter((m) => m.status === "loaded");
  select.innerHTML = "";
  for (const m of usable) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.name;
    if (health && m.name === health.defaults.model) opt.selected = true;
    select.append(opt);
  }
  if (!usable.length) {
    select.innerHTML = '<option value="">no models loaded</option>';
  }
}

// ------------------------------------------------------------------ submit

el("submit-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = el("submit");
  const errorBox = el("form-error");
  errorBox.hidden = true;
  button.disabled = true;

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: el("url").value.trim(),
        model: el("model").value,
        output_format: el("format").value,
      }),
    });
    const body = await response.json();

    if (!response.ok) {
      errorBox.textContent = body.detail || `Request failed (${response.status})`;
      errorBox.hidden = false;
      return;
    }

    // 200 means an existing job already covers this: a cached result, or one
    // already running. Either way we just show that job.
    upsertJob(body);
    if (!isTerminal(body.stage)) watch(body.id);
    el("url").value = "";
  } catch (err) {
    errorBox.textContent = String(err);
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
});

// -------------------------------------------------------------------- jobs

const isTerminal = (stage) => stage === "done" || stage === "failed";

async function loadRecent() {
  const { jobs } = await (await fetch("/api/jobs")).json();
  jobsEl.innerHTML = "";
  for (const job of jobs.reverse()) {
    upsertJob(job);
    if (!isTerminal(job.stage)) watch(job.id);
  }
  el("empty").hidden = jobs.length > 0;
}

function upsertJob(job) {
  let card = document.getElementById(`job-${job.id}`);
  if (!card) {
    card = document.createElement("article");
    card.className = "job";
    card.id = `job-${job.id}`;
    jobsEl.prepend(card);
  }
  card.replaceChildren(...renderJob(job));
  el("empty").hidden = true;
}

function renderJob(job) {
  const head = node("div", "job__head", [
    node("div", "job__title", job.display),
    node("div", "job__meta", metaLine(job)),
  ]);

  const reached = STAGES.indexOf(job.stage);
  const failed = job.stage === "failed";
  const strip = node("div", "stages");

  STAGES.forEach((stage, index) => {
    const seg = node("div", "stage");
    if (failed && index <= Math.max(reached, 0)) {
      seg.classList.add("stage--failed");
    } else if (index < reached) {
      seg.classList.add("stage--done");
    } else if (stage === job.stage && MEASURED.includes(stage) && job.progress > 0) {
      seg.classList.add("stage--active");
      const fill = document.createElement("i");
      fill.style.setProperty("--pct", `${Math.round(job.progress * 100)}%`);
      seg.append(fill);
    } else if (stage === job.stage && MEASURED.includes(stage)) {
      // No fraction yet: a hi-res download arrives as segmented MPD with no
      // Content-Length, and a separation reports nothing until its first
      // chunk. Sweep rather than sit at a frozen 0%, which reads as hung.
      seg.classList.add("stage--busy");
    } else if (job.stage === "done") {
      seg.classList.add("stage--done");
    }
    strip.append(seg);
  });

  const labels = node("div", "stage-labels",
    STAGES.map((s) => node("span", "", s)));

  const parts = [head, strip, labels];

  if (job.error) parts.push(node("div", "job__error", job.error));
  if (job.stage === "done") parts.push(stems(job));
  return parts;
}

function metaLine(job) {
  const bits = [];
  if (MEASURED.includes(job.stage) && job.progress > 0) {
    bits.push(`${job.stage} ${Math.round(job.progress * 100)}%`);
  } else {
    bits.push(job.stage);
  }
  if (job.duration) bits.push(formatDuration(job.duration));
  bits.push(job.model.replace(/\.(ckpt|onnx|pth|yaml)$/, ""));
  bits.push(job.output_format);
  return bits.join(" · ");
}

function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = String(seconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function stems(job) {
  const wrap = node("div", "stems");
  for (const which of ["vocals", "instrumental"]) {
    const url = `/api/jobs/${job.id}/stems/${which}`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = url;

    const link = document.createElement("a");
    link.href = url;
    link.download = "";
    link.textContent = "Download";

    wrap.append(node("div", "stem", [
      node("div", "stem__name", which),
      audio,
      link,
    ]));
  }
  return wrap;
}

function node(tag, className, children) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (typeof children === "string") element.textContent = children;
  else if (Array.isArray(children)) element.append(...children);
  return element;
}

// --------------------------------------------------------------------- sse

function watch(jobId) {
  if (streams.has(jobId)) return;
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  streams.set(jobId, source);

  source.addEventListener("update", (event) => {
    const job = JSON.parse(event.data);
    upsertJob(job);
    if (isTerminal(job.stage)) close(jobId);
  });
  // The server closes the stream on a terminal stage; EventSource would
  // reconnect forever, so treat an error on a finished job as expected.
  source.addEventListener("error", () => close(jobId));
  source.addEventListener("gone", () => {
    document.getElementById(`job-${jobId}`)?.remove();
    close(jobId);
  });
}

function close(jobId) {
  streams.get(jobId)?.close();
  streams.delete(jobId);
}

// -------------------------------------------------------------------- boot

(async function boot() {
  const health = await loadHealth();
  await loadModels(health);
  await loadRecent();
  // Cheap poll so the banner clears once you log in via the CLI.
  setInterval(loadHealth, 15000);
})();
