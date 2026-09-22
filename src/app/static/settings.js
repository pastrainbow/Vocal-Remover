"use strict";

const el = (id) => document.getElementById(id);

const parseModelLines = (text) =>
  text.split("\n").map((s) => s.trim()).filter(Boolean);

function rebuildDefaultModelOptions(names, selected) {
  const select = el("default-model");
  select.innerHTML = "";
  if (!names.length) {
    select.innerHTML = '<option value="">add a model above</option>';
    return;
  }
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === selected) opt.selected = true;
    select.append(opt);
  }
  // The previous default may have just been deleted from the textarea; fall
  // back to whatever is first rather than silently submitting nothing.
  if (!names.includes(selected)) select.value = names[0];
}

async function loadCurrent() {
  const settings = await (await fetch("/api/settings/models")).json();
  el("preload").value = settings.preload_models.join("\n");
  rebuildDefaultModelOptions(settings.preload_models, settings.default_model);
  el("format").value = settings.output_format;
  el("segment-size").value = settings.segment_size ?? "";
}

async function loadStatus() {
  const list = el("status-list");
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
    list.append(node("li", "", bits.join(" — ")));
  }
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

el("preload").addEventListener("input", () => {
  rebuildDefaultModelOptions(parseModelLines(el("preload").value),
                             el("default-model").value);
});

el("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = el("save");
  const errorBox = el("form-error");
  const okBox = el("form-ok");
  errorBox.hidden = true;
  okBox.hidden = true;
  button.disabled = true;

  const segmentRaw = el("segment-size").value.trim();
  const body = {
    preload_models: parseModelLines(el("preload").value),
    default_model: el("default-model").value,
    output_format: el("format").value,
    segment_size: segmentRaw ? Number(segmentRaw) : null,
  };

  try {
    const response = await fetch("/api/settings/models", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();

    if (!response.ok) {
      errorBox.textContent = result.detail || `Save failed (${response.status})`;
      errorBox.hidden = false;
      return;
    }

    okBox.textContent = "Saved. Restart the server to apply changes to " +
      "preloaded models or segment size.";
    okBox.hidden = false;
    rebuildDefaultModelOptions(result.preload_models, result.default_model);
  } catch (err) {
    errorBox.textContent = String(err);
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
  }
});

(async function boot() {
  await loadCurrent();
  await loadStatus();
})();
