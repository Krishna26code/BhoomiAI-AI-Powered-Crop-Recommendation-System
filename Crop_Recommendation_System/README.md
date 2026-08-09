# 🌾 Bhoomi — Crop Recommendation System

Flask frontend + RandomForest ML backend for crop recommendation, built on
top of Krishna's notebook (`Copy_of_Crop_Recommendation_System.ipynb`).

## Features
- 🤖 Farming Q&A chatbot (rule-based, works offline, no paid API)
- 📊 Analytics dashboard — how often each crop was recommended (Chart.js)
- 🌦️ Live weather widget (OpenWeatherMap free tier)
- 📋 Last-10 prediction history (browser localStorage)
- 🌍 Hindi / English toggle
- 📱 Installable PWA (manifest + service worker, works offline for the UI shell)
- 📄 One-click PDF recommendation report (ReportLab)
- 🗺️ Season-wise crop calendar for all 22 crops

## Setup

```bash
cd crop_app
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Re)train the model from the CSV in data/ (already included; only needed if you change the dataset)
python train_model.py

# Optional but recommended: free weather API key
# Get one at https://openweathermap.org/api (takes 2 minutes, free tier = 1000 calls/day)
export OWM_API_KEY="your_key_here"        # Windows: set OWM_API_KEY=your_key_here

python app.py
```

Open **http://127.0.0.1:5000**

## Project structure
```
crop_app/
├── app.py                  # Flask app — all routes/APIs
├── train_model.py          # Trains RandomForest, saves model+scalers+stats
├── requirements.txt
├── data/
│   ├── Crop_recommendation.csv
│   └── app.db               # auto-created SQLite (prediction history for analytics)
├── model/                   # auto-created by train_model.py
│   ├── crop_recommendation_model.pkl
│   ├── minmax_scaler.pkl
│   ├── standard_scaler.pkl
│   ├── crop_dict.json
│   └── crop_stats.json
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/app.js
    ├── manifest.json
    ├── service-worker.js
    └── icons/
```

## Notes / fixes vs the original notebook
- The notebook fit a `StandardScaler` on `x_train` but the `recommendation()`
  function fed **raw, unscaled** values into `rfc.predict()` — inconsistent
  with training. `train_model.py` now saves the scaler(s) and `app.py`
  applies the same MinMax → StandardScaler pipeline at inference time.
- Added `random_state=42` to `RandomForestClassifier` for reproducibility.

## Deploying for free
- **Render / Railway / PythonAnywhere free tier** all work well for a small
  Flask app like this. Just set the `OWM_API_KEY` environment variable in
  their dashboard — never commit it to git.
- For the PWA install prompt to work, the site needs to be served over
  **HTTPS** (or `localhost` for testing) — free hosts above give you HTTPS
  automatically.
