"use strict";

const el = (id) => document.getElementById(id);

let currentModel = null;
let currentSpec = null;  // the last GET/PUT response: {model, arch, load_time_fields, per_job_fields, kinds, values, applied}
//: What each input was drawn with, so submit can send only what changed.
//: Prefilling an inherited value and then saving the whole form would turn
//: "leave it to the model" into a pin on today's number, which is a
//: behaviour change nobody asked for by pressing Save.
let rendered = {};

function node(tag, className, children) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (typeof children === "string") element.textContent = children;
  else if (Array.isArray(children)) element.append(...children);
  return element;
}

const prettify = (name) =>
  name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

// --------------------------------------------------------------- model list

async function loadModelList() {
  const list = el("model-list");
  let models;
  try {
    ({ models } = await (await fetch("/api/models")).json());
  } catch {
    list.replaceChildren(node("li", "", "could not reach the server"));
    return;
  }
  list.innerHTML = "";
  if (!models.length) {
    list.append(node("li", "", "no models loaded"));
    return;
  }
  for (const m of models) {
    const bits = [m.name, m.status];
    if (m.device) bits.push(m.device);
    if (m.error) bits.push(m.error);

    const item = node("li", "status-list__item", bits.join(" — "));
    item.tabIndex = 0;
    item.addEventListener("click", () => selectModel(m.name));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectModel(m.name);
      }
    });
    if (m.name === currentModel) item.classList.add("status-list__item--selected");
    list.append(item);
  }
}

// ----------------------------------------------------------------- params

async function selectModel(name) {
  currentModel = name;
  await loadModelList();  // just to refresh the --selected highlight

  el("form-error").hidden = true;
  el("form-ok").hidden = true;

  const response = await fetch(`/api/models/${encodeURIComponent(name)}/params`);
  const body = await response.json();
  if (!response.ok) {
    el("form-error").textContent = body.detail || `Could not load params (${response.status})`;
    el("form-error").hidden = false;
    el("params-form").hidden = true;
    return;
  }

  currentSpec = body;
  renderForm(body);
}

/** One <label class="field"> per param, built entirely from what the API
 *  says about this model - no per-architecture table in this file. `kind`
 *  ("bool" | "text" | "number") picks the input type; membership in
 *  load_time_fields vs per_job_fields picks the "(restart)" marker.
 *
 *  A saved null means "leave it to the model", which is not something to
 *  show as an empty box - `applied` carries what the loaded model actually
 *  resolved it to, so that is what goes in the field, noted as inherited. */
function renderForm(spec) {
  el("params-heading").textContent = `${spec.model} (${spec.arch})`;

  const fields = el("params-fields");
  fields.innerHTML = "";
  const restartFields = new Set(spec.load_time_fields);
  const applied = spec.applied || {};
  rendered = {};

  for (const name of [...spec.load_time_fields, ...spec.per_job_fields]) {
    const kind = spec.kinds[name];
    const saved = spec.values[name];
    const inherited = saved === null || saved === undefined;
    // Nothing to inherit when the model is not loaded: leave it blank and
    // let the placeholder say so rather than invent a number.
    const shown = inherited && applied[name] !== undefined && applied[name] !== null
      ? applied[name]
      : saved;

    const input = document.createElement("input");
    input.id = `field-${name}`;
    input.name = name;

    if (kind === "bool") {
      input.type = "checkbox";
      input.checked = Boolean(shown);
    } else if (kind === "number") {
      input.type = "number";
      input.step = "any";
      input.placeholder = "model default";
      input.value = shown === null || shown === undefined ? "" : shown;
    } else {
      input.type = "text";
      input.value = shown ?? "";
    }

    rendered[name] = readInput(input, kind);

    const labelText = prettify(name) + (restartFields.has(name) ? " (restart)" : "");
    const parts = [node("span", "", labelText), input];
    if (inherited && shown !== null && shown !== undefined) {
      const note = node("span", "field__note", "from the model");
      // Stops the note describing a value the user has since replaced.
      input.addEventListener("input", () => {
        note.hidden = !same(readInput(input, kind), rendered[name]);
      });
      parts.push(note);
    }
    fields.append(node("label", "field", parts));
  }

  el("params-form").hidden = false;
}

function readInput(input, kind) {
  if (kind === "bool") return input.checked;
  if (kind === "number") {
    const raw = input.value.trim();
    return raw === "" ? null : Number(raw);
  }
  return input.value;
}

const same = (a, b) => a === b || (a === null && b === null);

/** Only what the user actually changed. The API merges a partial body, so
 *  anything left alone keeps its stored state - in particular an inherited
 *  field stays inherited rather than being pinned to the number shown. */
function changedFields(spec) {
  const body = {};
  for (const name of [...spec.load_time_fields, ...spec.per_job_fields]) {
    const value = readInput(el(`field-${name}`), spec.kinds[name]);
    if (!same(value, rendered[name])) body[name] = value;
  }
  return body;
}

el("params-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentModel || !currentSpec) return;

  const button = el("save");
  const errorBox = el("form-error");
  const okBox = el("form-ok");
  errorBox.hidden = true;
  okBox.hidden = true;
  button.disabled = true;

  try {
    const response = await fetch(
      `/api/models/${encodeURIComponent(currentModel)}/params`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changedFields(currentSpec)),
      },
    );
    const result = await response.json();

    if (!response.ok) {
      errorBox.textContent = result.detail || `Save failed (${response.status})`;
      errorBox.hidden = false;
      return;
    }

    okBox.textContent = "Saved.";
    okBox.hidden = false;
    currentSpec = result;
    renderForm(result);
  } catch (err) {
    errorBox.textContent = String(err);
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
});

(async function boot() {
  await loadModelList();
})();
