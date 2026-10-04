/* =====================================================================
   Glucentra — Frontend
   Sends 17 feature inputs to the Python FastAPI backend at /predict.
   No client-side risk calculation is performed.
   ===================================================================== */

const API_BASE_URL = "http://localhost:8000";
const PREDICT_PATH = "/predict";

// ---------- Field schema ----------
const yesNo = [
  { value: 0, label: "No" },
  { value: 1, label: "Yes" },
];

const fields = [
  {
    name: "BMI",
    label: "BMI",
    type: "number",
    min: 10,
    max: 80,
    step: 0.1,
    hint: "Body mass index (kg/m²)",
  },
  { name: "Smoker", label: "Smoker", type: "select", options: yesNo },
  {
    name: "PhysActivity",
    label: "Physical activity (last 30 days)",
    type: "select",
    options: yesNo,
  },
  {
    name: "Fruits",
    label: "Eats fruit ≥ 1×/day",
    type: "select",
    options: yesNo,
  },
  {
    name: "Veggies",
    label: "Eats vegetables ≥ 1×/day",
    type: "select",
    options: yesNo,
  },
  {
    name: "HvyAlcoholConsump",
    label: "Heavy alcohol consumption",
    type: "select",
    options: yesNo,
  },
  {
    name: "GenHlth",
    label: "General health",
    type: "select",
    options: [
      { value: 1, label: "Excellent" },
      { value: 2, label: "Very good" },
      { value: 3, label: "Good" },
      { value: 4, label: "Fair" },
      { value: 5, label: "Poor" },
    ],
  },
  {
    name: "MentHlth",
    label: "Poor mental health days (0–30)",
    type: "number",
    min: 0,
    max: 30,
    step: 1,
  },
  {
    name: "PhysHlth",
    label: "Poor physical health days (0–30)",
    type: "number",
    min: 0,
    max: 30,
    step: 1,
  },
  {
    name: "DiffWalk",
    label: "Difficulty walking",
    type: "select",
    options: yesNo,
  },
  {
    name: "Sex",
    label: "Sex",
    type: "select",
    options: [
      { value: 0, label: "Female" },
      { value: 1, label: "Male" },
    ],
  },
  {
    name: "Age",
    label: "Age category",
    type: "select",
    options: [
      { value: 1, label: "18–24" },
      { value: 2, label: "25–29" },
      { value: 3, label: "30–34" },
      { value: 4, label: "35–39" },
      { value: 5, label: "40–44" },
      { value: 6, label: "45–49" },
      { value: 7, label: "50–54" },
      { value: 8, label: "55–59" },
      { value: 9, label: "60–64" },
      { value: 10, label: "65–69" },
      { value: 11, label: "70–74" },
      { value: 12, label: "75–79" },
      { value: 13, label: "80+" },
    ],
  },
  {
    name: "Education",
    label: "Education level",
    type: "select",
    options: [
      { value: 1, label: "Never attended" },
      { value: 2, label: "Elementary" },
      { value: 3, label: "Some high school" },
      { value: 4, label: "High school graduate" },
      { value: 5, label: "Some college" },
      { value: 6, label: "College graduate" },
    ],
  },
  {
    name: "Income",
    label: "Household income",
    type: "select",
    options: [
      { value: 1, label: "< $10k" },
      { value: 2, label: "$10–15k" },
      { value: 3, label: "$15–20k" },
      { value: 4, label: "$20–25k" },
      { value: 5, label: "$25–35k" },
      { value: 6, label: "$35–50k" },
      { value: 7, label: "$50–75k" },
      { value: 8, label: "$75k+" },
    ],
  },
  {
    name: "Stroke",
    label: "History of stroke",
    type: "select",
    options: yesNo,
  },
  {
    name: "HeartDiseaseorAttack",
    label: "Heart disease or attack",
    type: "select",
    options: yesNo,
  },
  {
    name: "FamilyHistory",
    label: "Family history of diabetes",
    type: "select",
    options: [
      { value: "none", label: "No family history" },
      { value: "one_parent", label: "One parent diabetic" },
      { value: "both_parents", label: "Both parents diabetic" },
      { value: "sibling", label: "Sibling diabetic" },
      { value: "extended", label: "Grandparent / Uncle / Aunt" },
    ],
  },
];

const defaults = {
  BMI: 27,
  Smoker: 0,
  PhysActivity: 1,
  Fruits: 1,
  Veggies: 1,
  HvyAlcoholConsump: 0,
  GenHlth: 3,
  MentHlth: 2,
  PhysHlth: 2,
  DiffWalk: 0,
  Sex: 0,
  Age: 7,
  Education: 5,
  Income: 6,
  Stroke: 0,
  HeartDiseaseorAttack: 0,
  FamilyHistory: "none",
};

let state = { ...defaults };
const history = [];

// ---------- Render form fields ----------
function renderFields() {
  const root = document.getElementById("fields");
  root.innerHTML = fields
    .map((f) => {
      if (f.type === "number") {
        return `
        <div class="field">
          <label for="${f.name}">${f.label}</label>
          <input id="${f.name}" name="${f.name}" type="number"
                 min="${f.min ?? ""}" max="${f.max ?? ""}" step="${f.step ?? ""}"
                 value="${state[f.name]}" />
          ${f.hint ? `<span class="hint">${f.hint}</span>` : ""}
        </div>`;
      }
      const opts = f.options
        .map(
          (o) =>
            `<option value="${o.value}" ${String(state[f.name]) === String(o.value) ? "selected" : ""}>${o.label}</option>`,
        )
        .join("");
      return `
      <div class="field">
        <label for="${f.name}">${f.label}</label>
        <select id="${f.name}" name="${f.name}">${opts}</select>
      </div>`;
    })
    .join("");

  fields.forEach((f) => {
    const el = document.getElementById(f.name);
    el.addEventListener("input", () => {
      let v = el.value;
      if (f.type === "number") v = v === "" ? 0 : Number(v);
      else if (f.name !== "FamilyHistory") v = Number(v);
      state[f.name] = v;
    });
  });
}

// ---------- Backend call ----------
async function callPredict(payload) {
  const res = await fetch(API_BASE_URL + PREDICT_PATH, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = j.detail || j.message || "";
    } catch {}
    throw new Error(detail || `Backend responded ${res.status}`);
  }
  return res.json();
}

// ---------- Helpers ----------
function pct(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function riskMeta(level) {
  if (level <= 1) return { label: "Low risk", cls: "risk-low" };
  if (level === 2) return { label: "Moderate risk", cls: "risk-mod" };
  return { label: "High risk", cls: "risk-high" };
}

function gaugeColor(v) {
  if (v < 25) return "#16a34a";
  if (v < 50) return "#d97706";
  return "#dc2626";
}

// ---------- Render results ----------
function renderResults(result) {
  const v = pct(result.combined_risk ?? result.final_risk);
  const level = result.risk_level ?? (v < 25 ? 1 : v < 50 ? 2 : 3);
  const meta = riskMeta(level);
  const radius = 70;
  const circumference = Math.PI * radius;
  const offset = circumference - (v / 100) * circumference;
  const color = gaugeColor(v);

  const lifestyle = pct(result.lifestyle_risk);
  const hered =
    typeof result.hereditary_mult === "number" ? result.hereditary_mult : 1;
  const heredLabel = result.hereditary_label || "";

  // Build suggestions HTML
  const suggestions = Array.isArray(result.suggestions)
    ? result.suggestions
    : [];
  const suggestionsHtml =
    suggestions.length > 0
      ? `
    <div class="suggestions">
      <p class="suggestions-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        Personalised suggestions
      </p>
      <ul class="suggestions-list">
        ${suggestions.map((s) => `<li>${s}</li>`).join("")}
      </ul>
    </div>`
      : "";

  const el = document.getElementById("results");
  el.classList.remove("empty");
  el.innerHTML = `
    <div class="results-head">
      <div>
        <h3>Risk results</h3>
        <p>From backend <code>/predict</code> response</p>
      </div>
      <span class="risk-pill ${meta.cls}">${meta.label}</span>
    </div>

    <div class="gauge-wrap">
      <svg viewBox="0 0 180 110">
        <path d="M 20 100 A 70 70 0 0 1 160 100" fill="none" stroke="#e7eaf0" stroke-width="14" stroke-linecap="round"/>
        <path d="M 20 100 A 70 70 0 0 1 160 100" fill="none" stroke="${color}" stroke-width="14"
              stroke-linecap="round" stroke-dasharray="${circumference}" stroke-dashoffset="${offset}"/>
      </svg>
      <div class="gauge-text">
        <span class="gauge-value">${v.toFixed(1)}%</span>
        <span class="gauge-label">Final risk</span>
      </div>
    </div>

    <div class="bars">
      <div class="bar-row">
        <div class="bar-top"><span class="name">Lifestyle risk</span><span class="val">${lifestyle.toFixed(1)}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${lifestyle}%;background:#0ea5e9"></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-top"><span class="name">Combined risk</span><span class="val">${v.toFixed(1)}%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${v}%;background:#2563eb"></div></div>
      </div>
      <div class="kv">
        <span class="name muted">Hereditary multiplier</span>
        <span class="v">×${hered.toFixed(2)} <span class="kv-sub">${heredLabel}</span></span>
      </div>
    </div>

    ${suggestionsHtml}

    <p class="disclaimer">This tool is intended for educational and research purposes only and does not replace professional medical diagnosis.</p>
  `;
}

function showError(msg) {
  const e = document.getElementById("error");
  e.textContent = msg;
  e.classList.remove("hidden");
}
function clearError() {
  document.getElementById("error").classList.add("hidden");
}

// ---------- History ----------
function familyLabel(v) {
  const f = fields.find((x) => x.name === "FamilyHistory");
  return f.options.find((o) => o.value === v)?.label ?? v;
}
function ageLabel(v) {
  const f = fields.find((x) => x.name === "Age");
  return f.options.find((o) => Number(o.value) === Number(v))?.label ?? v;
}
function riskBadge(level, v) {
  const l = level ?? (v < 25 ? 1 : v < 50 ? 2 : 3);
  if (l <= 1) return `<span class="risk-pill risk-low small">Low</span>`;
  if (l === 2) return `<span class="risk-pill risk-mod small">Moderate</span>`;
  return `<span class="risk-pill risk-high small">High</span>`;
}

function renderHistory() {
  const tbody = document.getElementById("history-body");
  if (history.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="6">No predictions yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = history
    .map((h) => {
      const v = pct(h.result.combined_risk ?? h.result.final_risk);
      return `<tr>
      <td>${new Date(h.ts).toLocaleTimeString()}</td>
      <td>${h.input.BMI}</td>
      <td>${ageLabel(h.input.Age)}</td>
      <td>${familyLabel(h.input.FamilyHistory)}</td>
      <td class="num"><strong>${v.toFixed(1)}%</strong></td>
      <td>${riskBadge(h.result.risk_level, v)}</td>
    </tr>`;
    })
    .join("");
}

// ---------- Submit ----------
async function onSubmit(e) {
  e.preventDefault();
  clearError();
  const btn = document.getElementById("submit-btn");
  const label = btn.querySelector(".btn-label");
  btn.disabled = true;
  label.innerHTML = `<span class="spinner"></span> Predicting…`;
  try {
    const result = await callPredict(state);
    renderResults(result);
    history.unshift({ ts: Date.now(), input: { ...state }, result });
    if (history.length > 25) history.length = 25;
    renderHistory();

    // Scroll results into view on mobile
    document
      .getElementById("results")
      .scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showError(
      err.message ||
        "Failed to reach the prediction backend. Make sure the Python API is running at http://localhost:8000.",
    );
  } finally {
    btn.disabled = false;
    label.textContent = "Run prediction";
  }
}

function onReset() {
  state = { ...defaults };
  renderFields();
  // Reset results panel back to empty state
  const el = document.getElementById("results");
  el.classList.add("empty");
  el.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">i</div>
      <h3>Awaiting prediction</h3>
      <p class="muted">Fill in patient inputs and run the model. Results will appear here.</p>
    </div>`;
}

// ---------- Init ----------
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("year").textContent = new Date().getFullYear();
  document.getElementById("endpoint-display").textContent =
    `POST ${API_BASE_URL}${PREDICT_PATH}`;
  renderFields();
  renderHistory();
  document.getElementById("predict-form").addEventListener("submit", onSubmit);
  document.getElementById("reset-btn").addEventListener("click", onReset);
  document.getElementById("clear-history").addEventListener("click", () => {
    history.length = 0;
    renderHistory();
  });
});
