"""Campus Placement Predictor - single-file Flask app (UI + CSS + JS included).

Keep decision_tree_algo.pkl in the same folder as this file."""
import os
import pickle

import pandas as pd
from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_algo.pkl")

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Placement Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    /* Colour palette */
    --ink:    #1E1B4B;   /* deep indigo  - side panel, headings */
    --violet: #6D5BF7;   /* primary      - sliders, selected options, button */
    --mint:   #2DD4BF;   /* positive     - "placed" result */
    --coral:  #FF7A66;   /* attention    - "not yet" result */
    --mist:   #F4F2FF;   /* page background */
    --card:   #FFFFFF;
    --text:   #2B2A44;
    --muted:  #6F6D8F;
    --line:   #E1DEF5;
    --panel:  var(--ink);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    font-family: "Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--mist);
    color: var(--text);
    line-height: 1.5;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(300px, 420px) 1fr;
    min-height: 100vh;
  }

  /* ---------- Result panel ---------- */
  aside {
    background: var(--panel);
    color: #fff;
    padding: 48px 40px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    transition: background .5s ease;
  }
  aside[data-state="placed"]  { --panel: #0F5E57; }
  aside[data-state="not"]     { --panel: #7A2E3B; }
  .brand { font-weight: 800; font-size: 1.05rem; letter-spacing: .2px; display: flex; align-items: center; gap: 10px; }
  .brand i { width: 14px; height: 14px; border-radius: 4px; background: var(--mint); display: inline-block; transform: rotate(45deg); }
  aside h1 { font-size: 2.3rem; line-height: 1.12; margin: 56px 0 14px; font-weight: 800; }
  aside p.lead { color: #C9C6F0; max-width: 32ch; margin: 0; }

  .result { margin-top: auto; padding-top: 28px; min-height: 260px; }
  .result .verdict { font-size: 2rem; font-weight: 800; line-height: 1.15; margin: 0 0 8px; }
  .result .sub { color: rgba(255,255,255,.8); margin: 0 0 24px; }
  .badge {
    width: 64px; height: 64px; border-radius: 50%;
    display: grid; place-items: center; font-size: 1.9rem; font-weight: 800;
    margin-bottom: 18px;
  }
  [data-state="placed"] .badge { background: var(--mint); color: #063B36; }
  [data-state="not"]    .badge { background: var(--coral); color: #4A1119; }
  .factors h3 { font-size: .9rem; font-weight: 600; color: rgba(255,255,255,.75); margin: 0 0 10px; }
  .bar { display: grid; grid-template-columns: 130px 1fr 40px; align-items: center; gap: 10px; font-size: .85rem; margin: 6px 0; }
  .bar span.track { height: 8px; border-radius: 99px; background: rgba(255,255,255,.18); overflow: hidden; }
  .bar span.fill  { display: block; height: 100%; width: 0; border-radius: 99px; background: #fff; transition: width .9s cubic-bezier(.2,.8,.2,1); }
  .bar b { text-align: right; font-weight: 600; }
  .empty { color: #C9C6F0; border: 1.5px dashed rgba(255,255,255,.25); border-radius: 16px; padding: 22px; }
  .hidden { display: none; }

  /* ---------- Form ---------- */
  main { padding: 48px clamp(20px, 5vw, 72px); max-width: 860px; }
  main h2 { font-size: 1.5rem; margin: 0 0 4px; color: var(--ink); }
  main > p { margin: 0 0 28px; color: var(--muted); }
  fieldset {
    border: 0; margin: 0 0 20px; padding: 24px 26px;
    background: var(--card); border-radius: 20px;
    box-shadow: 0 1px 0 var(--line), 0 12px 32px -18px rgba(30,27,75,.25);
  }
  legend { float: left; width: 100%; font-weight: 800; color: var(--ink); font-size: 1.05rem; padding: 0; margin-bottom: 16px; }
  fieldset > *:not(legend) { clear: both; }

  .slider { margin-bottom: 18px; }
  .slider .row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; font-weight: 500; }
  .slider input[type=number] {
    width: 76px; padding: 6px 8px; font: inherit; font-weight: 600; color: var(--ink);
    border: 1.5px solid var(--line); border-radius: 10px; text-align: right; background: var(--mist);
  }
  input[type=range] { width: 100%; accent-color: var(--violet); height: 6px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px 24px; }
  .field > label, .field > .lab { display: block; font-weight: 500; margin-bottom: 8px; }
  .seg { display: inline-flex; flex-wrap: wrap; gap: 6px; padding: 4px; background: var(--mist); border-radius: 12px; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg label {
    padding: 8px 14px; border-radius: 9px; cursor: pointer; font-weight: 600; font-size: .9rem; color: var(--muted);
    transition: background .15s, color .15s;
  }
  .seg input:checked + label { background: var(--violet); color: #fff; }
  .seg input:focus-visible + label { outline: 3px solid var(--mint); outline-offset: 2px; }
  .seg label:hover { color: var(--ink); }
  .seg input:checked + label:hover { color: #fff; }

  button.go {
    width: 100%; margin-top: 8px; padding: 16px; border: 0; border-radius: 16px;
    background: var(--violet); color: #fff; font: inherit; font-weight: 800; font-size: 1.05rem; cursor: pointer;
    box-shadow: 0 14px 28px -14px var(--violet);
    transition: transform .12s, background .2s;
  }
  button.go:hover { background: #5A47E8; }
  button.go:active { transform: translateY(1px); }
  button.go:focus-visible { outline: 3px solid var(--mint); outline-offset: 3px; }
  button.go[disabled] { opacity: .65; cursor: progress; }
  .err { color: #B3261E; font-weight: 600; margin: 12px 0 0; min-height: 1.4em; }

  @media (max-width: 880px) {
    .layout { grid-template-columns: 1fr; }
    aside { position: static; height: auto; padding: 32px 24px; }
    aside h1 { margin-top: 28px; font-size: 1.9rem; }
    .result { margin-top: 28px; min-height: 0; }
    main { padding: 28px 18px 48px; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="layout">

  <aside id="panel" data-state="idle" aria-live="polite">
    <div class="brand"><i></i> Placement Predictor</div>
    <h1>Will your profile get placed?</h1>
    <p class="lead">Enter your academic scores and background. A decision tree trained on campus placement records gives you a verdict.</p>

    <div class="result">
      <div id="empty" class="empty">Your result will show up here after you press <b>Predict placement</b>.</div>
      <div id="out" class="hidden">
        <div class="badge" id="badge"></div>
        <p class="verdict" id="verdict"></p>
        <p class="sub" id="sub"></p>
        <div class="factors">
          <h3>What the model weighs most</h3>
          <div id="factors"></div>
        </div>
      </div>
    </div>
  </aside>

  <main>
    <h2>Your profile</h2>
    <p>All percentages are out of 100.</p>

    <form id="form" novalidate>
      <fieldset>
        <legend>Scores</legend>
        <div id="sliders"></div>
      </fieldset>

      <fieldset>
        <legend>Background</legend>
        <div class="grid" id="background"></div>
      </fieldset>

      <fieldset>
        <legend>Experience and MBA</legend>
        <div class="grid" id="experience"></div>
      </fieldset>

      <button class="go" id="btn" type="submit">Predict placement</button>
      <p class="err" id="err" role="alert"></p>
    </form>
  </main>
</div>

<script>
  const SLIDERS = [
    ["ssc_p",   "10th percentage",       65],
    ["hsc_p",   "12th percentage",       65],
    ["degree_p","Degree percentage",     65],
    ["etest_p", "Employability test %",  70],
    ["mba_p",   "MBA percentage",        62],
  ];
  const CHOICES = {
    background: [
      ["gender", "Gender",       [["M","Male"],["F","Female"]]],
      ["ssc_b",  "10th board",   [["Central","Central"],["Others","Others"]]],
      ["hsc_b",  "12th board",   [["Central","Central"],["Others","Others"]]],
      ["hsc_s",  "12th stream",  [["Commerce","Commerce"],["Science","Science"],["Arts","Arts"]]],
      ["degree_t","Degree type", [["Sci&Tech","Sci & Tech"],["Comm&Mgmt","Comm & Mgmt"],["Others","Others"]]],
    ],
    experience: [
      ["workex", "Work experience", [["No","No"],["Yes","Yes"]]],
      ["specialisation", "MBA specialisation", [["Mkt&Fin","Marketing & Finance"],["Mkt&HR","Marketing & HR"]]],
    ],
  };

  // Build sliders
  const sl = document.getElementById("sliders");
  SLIDERS.forEach(([id, label, def]) => {
    sl.insertAdjacentHTML("beforeend", `
      <div class="slider">
        <div class="row"><label for="${id}_n">${label}</label>
          <input type="number" id="${id}_n" min="0" max="100" step="0.01" value="${def}"></div>
        <input type="range" id="${id}" name="${id}" min="0" max="100" step="0.5" value="${def}" aria-label="${label}">
      </div>`);
    const r = document.getElementById(id), n = document.getElementById(id + "_n");
    r.addEventListener("input", () => n.value = r.value);
    n.addEventListener("input", () => r.value = n.value);
  });

  // Build segmented choices
  for (const [target, fields] of Object.entries(CHOICES)) {
    const box = document.getElementById(target);
    fields.forEach(([name, label, opts]) => {
      const items = opts.map(([v, t], i) =>
        `<input type="radio" name="${name}" id="${name}_${i}" value="${v}" ${i === 0 ? "checked" : ""}>
         <label for="${name}_${i}">${t}</label>`).join("");
      box.insertAdjacentHTML("beforeend",
        `<div class="field"><span class="lab">${label}</span><div class="seg" role="radiogroup" aria-label="${label}">${items}</div></div>`);
    });
  }

  // Submit
  const form = document.getElementById("form"), btn = document.getElementById("btn"), err = document.getElementById("err");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    err.textContent = "";
    const payload = {};
    SLIDERS.forEach(([id]) => payload[id] = document.getElementById(id + "_n").value);
    new FormData(form).forEach((v, k) => { if (!(k in payload)) payload[k] = v; });

    btn.disabled = true; btn.textContent = "Predicting…";
    try {
      const res = await fetch("/predict", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      show(data);
    } catch (ex) {
      err.textContent = ex.message;
    } finally {
      btn.disabled = false; btn.textContent = "Predict placement";
    }
  });

  function show({placed, factors}) {
    const panel = document.getElementById("panel");
    panel.dataset.state = placed ? "placed" : "not";
    document.getElementById("empty").classList.add("hidden");
    document.getElementById("out").classList.remove("hidden");
    document.getElementById("badge").textContent = placed ? "✓" : "✕";
    document.getElementById("verdict").textContent = placed ? "Placed" : "Not Placed";
    document.getElementById("sub").textContent = placed
      ? "Your scores and background match students who were placed."
      : "Your profile matches students who were not placed. Improving the top-weighted scores below helps most.";
    const max = Math.max(...factors.map(f => f.weight));
    const box = document.getElementById("factors");
    box.innerHTML = factors.map(f =>
      `<div class="bar"><span>${f.name}</span><span class="track"><span class="fill" data-w="${(f.weight / max) * 100}"></span></span><b>${f.weight}%</b></div>`
    ).join("");
    requestAnimationFrame(() => requestAnimationFrame(() =>
      box.querySelectorAll(".fill").forEach(el => el.style.width = el.dataset.w + "%")));
    if (window.innerWidth <= 880) panel.scrollIntoView({behavior: "smooth"});
  }
</script>
</body>
</html>
"""

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Categorical values were label-encoded alphabetically when the model was trained
# (e.g. LabelEncoder). If your training code used a different encoding, edit here.
CATEGORICAL = {
    "gender": {"F": 0, "M": 1},
    "ssc_b": {"Central": 0, "Others": 1},
    "hsc_b": {"Central": 0, "Others": 1},
    "hsc_s": {"Arts": 0, "Commerce": 1, "Science": 2},
    "degree_t": {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2},
    "workex": {"No": 0, "Yes": 1},
    "specialisation": {"Mkt&Fin": 0, "Mkt&HR": 1},
}
PERCENT_FIELDS = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

LABELS = {
    "ssc_p": "10th %",
    "hsc_p": "12th %",
    "degree_p": "Degree %",
    "etest_p": "Employability test %",
    "mba_p": "MBA %",
    "workex": "Work experience",
    "specialisation": "Specialisation",
    "gender": "Gender",
    "ssc_b": "10th board",
    "hsc_b": "12th board",
    "hsc_s": "12th stream",
    "degree_t": "Degree type",
}


def top_factors(n=5):
    pairs = sorted(zip(FEATURES, model.feature_importances_), key=lambda p: -p[1])
    return [
        {"name": LABELS.get(k, k), "weight": round(float(v) * 100, 1)}
        for k, v in pairs[:n]
        if v > 0
    ]


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    row = {}
    try:
        for field in PERCENT_FIELDS:
            value = float(data[field])
            if not 0 <= value <= 100:
                raise ValueError(f"{LABELS[field]} must be between 0 and 100.")
            row[field] = value
        for field, mapping in CATEGORICAL.items():
            if data[field] not in mapping:
                raise ValueError(f"Invalid choice for {LABELS[field]}.")
            row[field] = mapping[data[field]]
    except KeyError as e:
        return jsonify(error=f"Missing field: {e.args[0]}"), 400
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e) or "Please enter valid numbers."), 400

    X = pd.DataFrame([row])[FEATURES]
    placed = int(model.predict(X)[0])
    return jsonify(placed=bool(placed), factors=top_factors())


if __name__ == "__main__":
    app.run(debug=True, port=5000)