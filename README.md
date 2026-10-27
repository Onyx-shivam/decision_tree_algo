# decision_tree_algo

# Campus Placement Predictor

A small Flask web app that serves a trained **Decision Tree classifier** and tells a student whether their profile is predicted as **Placed** or **Not Placed**. The whole app (backend, HTML, CSS and JavaScript) lives in a single file, `app.py`.

## Features

- Single-file Flask app: just `app.py` and `requirements.txt` (plus the model file)
- Colourful, responsive UI with a split layout. The side panel changes colour with the result: teal for Placed, deep red for Not Placed
- Sliders linked to number boxes for scores, and toggle buttons for categorical choices
- Predictions run without a page reload through a JSON endpoint (`POST /predict`)
- Shows which features the model weighs most, taken from `feature_importances_`
- Server-side validation (percentages must be between 0 and 100)

## Project structure

```
decision_tree_algo/
├── app.py                    # Flask backend + embedded UI (HTML/CSS/JS)
├── requirements.txt          # Python dependencies
├── decision_tree_algo.pkl    # Trained model (must sit next to app.py)
└── README.md
```

## Getting started

### 1. Requirements

- Python 3.9 or newer (developed and tested on 3.12)

### 2. Create a virtual environment (recommended)

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## The model

| Property | Value |
|---|---|
| Algorithm | `sklearn.tree.DecisionTreeClassifier` (criterion: gini, no max depth) |
| Input features | 12 |
| Target classes | `0` = Not Placed, `1` = Placed |
| Tree size | depth 8, 19 leaves |

### Input features

| Feature | Meaning | Input type | Encoding used by the app |
|---|---|---|---|
| `ssc_p` | 10th percentage | number 0-100 | as entered |
| `hsc_p` | 12th percentage | number 0-100 | as entered |
| `degree_p` | Degree percentage | number 0-100 | as entered |
| `etest_p` | Employability test percentage | number 0-100 | as entered |
| `mba_p` | MBA percentage | number 0-100 | as entered |
| `gender` | Gender | M / F | F=0, M=1 |
| `ssc_b` | 10th board | Central / Others | Central=0, Others=1 |
| `hsc_b` | 12th board | Central / Others | Central=0, Others=1 |
| `hsc_s` | 12th stream | Arts / Commerce / Science | Arts=0, Commerce=1, Science=2 |
| `degree_t` | Degree type | Comm&Mgmt / Others / Sci&Tech | Comm&Mgmt=0, Others=1, Sci&Tech=2 |
| `workex` | Work experience | No / Yes | No=0, Yes=1 |
| `specialisation` | MBA specialisation | Mkt&Fin / Mkt&HR | Mkt&Fin=0, Mkt&HR=1 |

> **Important:** the categorical encodings above assume alphabetical label encoding (for example scikit-learn's `LabelEncoder`) was used when the model was trained. If you encoded the columns differently, edit the `CATEGORICAL` dictionary at the top of `app.py` so it matches your training code.

## API

### `POST /predict`

Send JSON with all 12 fields. Percentages can be numbers or numeric strings, and categorical fields use the text labels from the table above.

```json
{
  "ssc_p": 80, "hsc_p": 80, "degree_p": 75, "etest_p": 85, "mba_p": 70,
  "gender": "M", "ssc_b": "Central", "hsc_b": "Central",
  "hsc_s": "Commerce", "degree_t": "Comm&Mgmt",
  "workex": "Yes", "specialisation": "Mkt&HR"
}
```

Response:

```json
{
  "placed": true,
  "factors": [{ "name": "10th %", "weight": 39.6 }, "..."]
}
```

Invalid or missing input returns HTTP `400` with `{ "error": "..." }`.

## Customising

- **Colours:** edit the `:root { ... }` block at the top of the CSS inside `app.py`. The palette is indigo `#1E1B4B`, violet `#6D5BF7`, mint `#2DD4BF`, coral `#FF7A66` and mist `#F4F2FF`.
- **Wording:** the "Placed" / "Not Placed" text is in the `show()` JavaScript function near the bottom of `app.py`.
- **Default slider values:** change the `SLIDERS` list in the JavaScript section.

## Notes and limitations

- **Verdict, not probability.** The tree has no depth limit, so its leaves are pure and `predict_proba` returns only 0 or 1. That is why the app shows a verdict rather than a percentage. To get meaningful probabilities, retrain with `max_depth` or `min_samples_leaf` set.
- **Some features have no effect.** This particular tree never splits on `gender`, `ssc_b`, `hsc_b`, `hsc_s`, `degree_t` or `etest_p`, so changing them does not change the prediction.
- **Educational use.** The model learns patterns from a small historical dataset. It is a demo, not a reliable guide to real hiring decisions.
- **Never unpickle untrusted files.** Loading a `.pkl` file can run arbitrary code. Only use model files you created or trust.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | Run `python -m pip install -r requirements.txt` with the same Python you use to run the app. |
| `InconsistentVersionWarning` when loading the model | The model was saved with a different scikit-learn version. Install the version you trained with (`pip show scikit-learn` in the training environment) and pin it in `requirements.txt`. |
| `FileNotFoundError: decision_tree_algo.pkl` | Keep the `.pkl` file in the same folder as `app.py`. |
| Red text in the HTML/JS part of `app.py` in VS Code | Normal. VS Code colours the whole embedded page as a Python string. It is not an error. |
| Port 5000 already in use | Change `port=5000` in the last line of `app.py`. |

## Deployment

`app.py` runs Flask's development server with `debug=True`, which is for local use only. For production, set `debug=False` and use a production server:

```bash
# Linux / macOS
gunicorn app:app

# Windows (gunicorn does not run on Windows)
pip install waitress
waitress-serve --listen=0.0.0.0:5000 app:app
```

## Tech stack

Python, Flask, scikit-learn, pandas, NumPy, plain HTML/CSS/JavaScript (no frontend framework).