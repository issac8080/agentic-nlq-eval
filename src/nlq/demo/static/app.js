const state = { questions: [], runs: [], rows: [], tier: "all", dataset: "all", questionId: null, rowId: null };

const systemsEl = document.querySelector("#systems");
const filtersEl = document.querySelector("#filters");
const questionList = document.querySelector("#question-list");
const questionDetail = document.querySelector("#question-detail");
const resultList = document.querySelector("#result-list");
const resultDetail = document.querySelector("#result-detail");
const runSelect = document.querySelector("#run-select");
const runNote = document.querySelector("#run-note");
const metrics = document.querySelector("#metrics");

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("is-on"));
    document.querySelectorAll(".panel").forEach((item) => item.classList.remove("is-on"));
    button.classList.add("is-on");
    document.querySelector("#" + button.dataset.panel).classList.add("is-on");
  });
});

runSelect.addEventListener("change", () => loadRun(runSelect.value));

boot();

async function boot() {
  const response = await fetch("/api/bootstrap");
  const data = await response.json();
  state.questions = data.questions;
  state.runs = data.runs;
  systemsEl.innerHTML = data.systems.map((system) => `
    <article class="system-card"><strong>${escapeHtml(system.name)}</strong><p>${escapeHtml(system.summary)}</p></article>
  `).join("");
  renderFilters();
  renderQuestions();
  renderRunSelect();
}

function renderFilters() {
  const tiers = ["all", ...new Set(state.questions.map((item) => String(item.tier)))];
  const datasets = ["all", ...new Set(state.questions.map((item) => item.dataset))];
  filtersEl.innerHTML = [
    ...tiers.map((tier) => chip("tier", tier, tier === "all" ? "All tiers" : "Tier " + tier)),
    ...datasets.map((dataset) => chip("dataset", dataset, dataset === "all" ? "All data" : dataset)),
  ].join("");
  filtersEl.querySelectorAll(".chip").forEach((button) => {
    button.addEventListener("click", () => {
      state[button.dataset.key] = button.dataset.value;
      renderFilters();
      renderQuestions();
    });
  });
}

function chip(key, value, label) {
  const on = state[key] === value ? " is-on" : "";
  return `<button class="chip${on}" type="button" data-key="${key}" data-value="${escapeHtml(value)}">${escapeHtml(label)}</button>`;
}

function visibleQuestions() {
  return state.questions.filter((item) => {
    const tierOk = state.tier === "all" || String(item.tier) === state.tier;
    const dataOk = state.dataset === "all" || item.dataset === state.dataset;
    return tierOk && dataOk;
  });
}

function renderQuestions() {
  const items = visibleQuestions();
  if (!items.some((item) => item.id === state.questionId)) state.questionId = items[0] ? items[0].id : null;
  questionList.innerHTML = items.map((item) => `
    <li><button type="button" class="${item.id === state.questionId ? "is-on" : ""}" data-id="${escapeHtml(item.id)}">
      <b>${escapeHtml(item.id)}</b>
      <small>Tier ${item.tier} · ${escapeHtml(item.dataset)}</small>
      ${escapeHtml(item.question)}
    </button></li>
  `).join("");
  questionList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.questionId = button.dataset.id;
      renderQuestions();
    });
  });
  const selected = items.find((item) => item.id === state.questionId);
  questionDetail.innerHTML = selected ? questionCard(selected) : `<p class="empty">No questions in this filter.</p>`;
}

function questionCard(item) {
  const values = JSON.stringify(item.gold_answer.values, null, 2);
  return `
    <div class="meta">
      <span class="tag">Tier ${item.tier}</span>
      <span class="tag">${escapeHtml(item.dataset)}</span>
      <span class="tag">${escapeHtml(item.reference_scale)}</span>
    </div>
    <h2>${escapeHtml(item.question)}</h2>
    <h3>Gold SQL</h3>
    <pre>${escapeHtml(item.gold_sql)}</pre>
    <h3>Reference answer</h3>
    <pre>${escapeHtml(values)}</pre>
  `;
}

function renderRunSelect() {
  if (!state.runs.length) {
    runSelect.innerHTML = `<option value="">No logs yet</option>`;
    runNote.textContent = "Run an experiment to fill this view. Logs are read from runs/*.jsonl.";
    resultList.innerHTML = "";
    metrics.innerHTML = "";
    resultDetail.innerHTML = `<p class="empty">No experiment log has been written. The question tab still shows the benchmark and its gold SQL.</p>`;
    return;
  }
  runSelect.innerHTML = state.runs.map((run) => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}</option>`).join("");
  loadRun(state.runs[0].name);
}

async function loadRun(name) {
  if (!name) return;
  const response = await fetch("/api/runs/" + encodeURIComponent(name));
  const data = await response.json();
  state.rows = (data.rows || []).filter((row) => !row.warmup);
  runNote.textContent = `${state.rows.length} measured runs in ${name}. Warm-up rows are hidden.`;
  renderMetrics();
  renderResults();
}

function renderMetrics() {
  const systems = [...new Set(state.rows.map((row) => row.system))];
  metrics.innerHTML = systems.map((system) => {
    const rows = state.rows.filter((row) => row.system === system);
    const accuracy = rate(rows.map((row) => row.accuracy.execution_accuracy));
    const faithful = rate(rows.map((row) => row.faithfulness.faithful));
    return `<div class="metric"><b>${pct(accuracy)}</b><span>${escapeHtml(system)} execution accuracy</span></div>
            <div class="metric"><b>${pct(faithful)}</b><span>${escapeHtml(system)} faithful</span></div>`;
  }).join("");
}

function renderResults() {
  if (!state.rows.length) {
    resultList.innerHTML = "";
    resultDetail.innerHTML = `<p class="empty">This log has no measured rows.</p>`;
    return;
  }
  if (!state.rows.some((row) => row.run_id === state.rowId)) state.rowId = state.rows[0].run_id;
  resultList.innerHTML = state.rows.map((row) => `
    <li><button type="button" class="${row.run_id === state.rowId ? "is-on" : ""}" data-id="${escapeHtml(row.run_id)}">
      <b>${escapeHtml(row.question_id)} · ${escapeHtml(row.system)}</b>
      <small>${escapeHtml(row.scale)} · repeat ${row.repeat}</small>
    </button></li>
  `).join("");
  resultList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.rowId = button.dataset.id;
      renderResults();
    });
  });
  const selected = state.rows.find((row) => row.run_id === state.rowId);
  resultDetail.innerHTML = selected ? resultCard(selected) : "";
}

function resultCard(row) {
  const correct = row.accuracy.execution_accuracy ? "good" : "bad";
  const faithful = row.faithfulness.faithful ? "good" : "bad";
  const latency = row.latency || {};
  return `
    <div class="meta">
      <span class="tag ${correct}">${row.accuracy.execution_accuracy ? "correct SQL result" : "incorrect SQL result"}</span>
      <span class="tag ${faithful}">${row.faithfulness.faithful ? "faithful" : "unfaithful"}</span>
      <span class="tag">${row.failed ? "failed" : "completed"}</span>
    </div>
    <h2>${escapeHtml(row.question_id)}</h2>
    <p>${escapeHtml(row.narrative || "No narrative was stored.")}</p>
    <p>End to end ${num(latency.end_to_end_s)}s · model ${num(latency.llm_time_s)}s · SQL ${num(latency.exec_time_s)}s · retries ${row.retries}</p>
    <h3>Predicted SQL</h3>
    <pre>${escapeHtml(row.sql || "")}</pre>
    <h3>Answer values</h3>
    <pre>${escapeHtml(JSON.stringify(row.answer_values, null, 2))}</pre>
  `;
}

function rate(values) {
  if (!values.length) return null;
  return values.filter(Boolean).length / values.length;
}

function pct(value) {
  return value == null ? "—" : Math.round(value * 100) + "%";
}

function num(value) {
  return Number(value || 0).toFixed(2);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}
