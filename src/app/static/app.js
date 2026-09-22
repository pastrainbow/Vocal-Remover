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
    // Clear a banner from an earlier poll: the worker can come back on its
    // own now, so "down" is a state the page has to be able to leave.
    el("worker-warning").hidden = true;
  } else {
    setPill("worker down", "error");
    const w = el("worker-warning");
    // The worker restarts itself, so `detail` is usually "restarting in 5s
    // after: ..." rather than something to act on. Show it either way: the
    // reason it went down is the useful part, and the one case that does
    // need a human says so itself.
    w.textContent = `The worker is not running - ${worker.detail}.`;
    w.hidden = false;
  }

  await renderAuth(health);
  // Nothing can be submitted without a Tidal session or a worker.
  el("submit").disabled = !health.tidal.authenticated || !worker.running;

  return health;
}

// ------------------------------------------------------------------- login

// How often the page asks where the login has got to. The server does the
// real polling on its own thread, at Tidal's interval, so this only paces
// the countdown on screen.
const LOGIN_POLL_MS = 2000;
let loginTimer = null;
let loginCommand = "";

async function renderAuth(health) {
  loginCommand = health.tidal.login_command;
  const box = el("auth-warning");
  el("logout").hidden = !health.tidal.authenticated;

  if (health.tidal.authenticated) {
    stopLoginPoll();
    box.replaceChildren();
    box.hidden = true;
    return;
  }

  box.hidden = false;
  // A login may already be running - started in another tab, or before this
  // page was reloaded - so ask before offering to start a new one.
  showLogin(await fetchLogin());
}

async function fetchLogin() {
  try {
    return await (await fetch("/api/login")).json();
  } catch {
    return { stage: "idle", detail: "" };
  }
}

function showLogin(login) {
  const box = el("auth-warning");

  if (login.stage === "pending") {
    box.replaceChildren(
      node("strong", "", "Finish signing in to Tidal:"),
      loginLink(login.verification_url),
      node("span", "", "then enter"),
      node("code", "", login.user_code),
      node("span", "banner__note",
           `expires in ${formatDuration(login.seconds_left)}`),
    );
    startLoginPoll();
    return;
  }

  stopLoginPoll();
  const failed = login.stage === "expired" || login.stage === "error";
  box.replaceChildren(
    node("strong", "", failed
      ? `Sign-in did not finish — ${login.detail}`
      : "Not signed in to Tidal."),
    loginButton(failed ? "Try again" : "Sign in to Tidal"),
    node("span", "banner__note", "or run:"),
    node("code", "", loginCommand),
  );
}

async function startLogin(event) {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "asking Tidal…";
  try {
    const response = await fetch("/api/login", { method: "POST" });
    const body = await response.json();
    showLogin(response.ok
      ? body
      : { stage: "error", detail: body.detail || `failed (${response.status})` });
  } catch (err) {
    showLogin({ stage: "error", detail: String(err) });
  }
}

function startLoginPoll() {
  if (loginTimer) return;
  loginTimer = setInterval(async () => {
    const login = await fetchLogin();
    if (login.stage === "ok") {
      stopLoginPoll();
      await loadHealth();  // clears the banner and enables Separate
      return;
    }
    showLogin(login);
  }, LOGIN_POLL_MS);
}

function stopLoginPoll() {
  if (loginTimer) clearInterval(loginTimer);
  loginTimer = null;
}

el("logout").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const errorBox = el("form-error");
  button.disabled = true;
  errorBox.hidden = true;
  try {
    const response = await fetch("/api/logout", { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      errorBox.textContent =
        `Sign-out failed: ${body.detail || response.status}`;
      errorBox.hidden = false;
    }
  } catch (err) {
    errorBox.textContent = `Sign-out failed: ${err}`;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    // Whatever happened, let the server say where things stand: this shows
    // the sign-in banner and disables Separate.
    await loadHealth();
  }
});

function loginLink(url) {
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  // The URL carries the code, so opening it is usually the whole step.
  a.textContent = "open the Tidal page";
  return a;
}

function loginButton(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", startLogin);
  return button;
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
  // Cheap poll: catches a session that expires while the page is open, or a
  // login done elsewhere (another tab, or the CLI).
  setInterval(loadHealth, 15000);
})();
