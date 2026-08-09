"""
Crop Recommendation System - Flask Backend
Author: Krishna Sharma

Features wired up here:
  - /api/predict         -> ML prediction (RandomForest, scaled inputs)
  - /api/analytics        -> aggregate recommendation counts for the dashboard
  - /api/chatbot           -> lightweight rule-based farming Q&A bot (FREE, no paid LLM API)
  - /api/weather           -> proxies OpenWeatherMap (keeps your API key server-side)
  - /api/report            -> generates a PDF recommendation report
  - /api/crop-info/<crop>  -> ideal condition ranges + calendar info for a crop
  - /api/crop-calendar     -> season-wise sowing calendar for all crops

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import json
import pickle
import sqlite3
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify, send_file, g

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "data", "app.db")

# ---- OpenWeatherMap ----------------------------------------------------
# Get a FREE key at https://openweathermap.org/api (free tier = 1000 calls/day)
# Put it in an environment variable so it never lands in your source code / git repo:
#   export OWM_API_KEY="your_key_here"        (Linux/Mac)
#   setx OWM_API_KEY "your_key_here"           (Windows)
OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# ---- Load model artifacts ------------------------------------------------
with open(os.path.join(MODEL_DIR, "crop_recommendation_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "minmax_scaler.pkl"), "rb") as f:
    minmax_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "standard_scaler.pkl"), "rb") as f:
    standard_scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "crop_dict.json"), "r") as f:
    _dicts = json.load(f)
    CROP_DICT = _dicts["crop_dict"]
    REVERSE_CROP_DICT = {int(k): v for k, v in _dicts["reverse_crop_dict"].items()}
    FEATURE_ORDER = _dicts["feature_order"]
with open(os.path.join(MODEL_DIR, "crop_stats.json"), "r") as f:
    CROP_STATS = json.load(f)

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "🧵",
    "jute": "🌿", "coffee": "☕",
}

CROP_CALENDAR = {
    "rice":        {"season": "Kharif", "sow": "June - July", "harvest": "Nov - Dec"},
    "maize":       {"season": "Kharif / Rabi", "sow": "June - July / Oct", "harvest": "Sept - Oct / Feb"},
    "chickpea":    {"season": "Rabi", "sow": "Oct - Nov", "harvest": "Feb - Mar"},
    "kidneybeans": {"season": "Kharif", "sow": "June - July", "harvest": "Sept - Oct"},
    "pigeonpeas":  {"season": "Kharif", "sow": "June - July", "harvest": "Dec - Jan"},
    "mothbeans":   {"season": "Kharif", "sow": "June - July", "harvest": "Sept - Oct"},
    "mungbean":    {"season": "Kharif / Zaid", "sow": "June - July / Mar", "harvest": "Sept / May"},
    "blackgram":   {"season": "Kharif / Rabi", "sow": "June - July / Oct", "harvest": "Sept / Feb"},
    "lentil":      {"season": "Rabi", "sow": "Oct - Nov", "harvest": "Feb - Mar"},
    "pomegranate": {"season": "Year-round (best: winter)", "sow": "Jan - Feb / July - Aug", "harvest": "5-7 months after flowering"},
    "banana":      {"season": "Year-round", "sow": "June - July / Oct - Nov", "harvest": "11-13 months after planting"},
    "mango":       {"season": "Year-round (perennial)", "sow": "July - Aug (planting)", "harvest": "Apr - Jun"},
    "grapes":      {"season": "Rabi (pruning based)", "sow": "Jan - Feb", "harvest": "Feb - Apr"},
    "watermelon":  {"season": "Zaid", "sow": "Feb - Mar", "harvest": "May - June"},
    "muskmelon":   {"season": "Zaid", "sow": "Feb - Mar", "harvest": "May - June"},
    "apple":       {"season": "Rabi (temperate, perennial)", "sow": "Dec - Feb (planting)", "harvest": "Aug - Oct"},
    "orange":      {"season": "Year-round (perennial)", "sow": "June - July", "harvest": "Nov - Jan"},
    "papaya":      {"season": "Year-round", "sow": "Feb - Mar / July - Aug", "harvest": "9-11 months after planting"},
    "coconut":     {"season": "Year-round (perennial)", "sow": "June - July", "harvest": "12 months a year (perennial)"},
    "cotton":      {"season": "Kharif", "sow": "Apr - May", "harvest": "Oct - Jan"},
    "jute":        {"season": "Kharif", "sow": "Mar - May", "harvest": "July - Sept"},
    "coffee":      {"season": "Year-round (perennial)", "sow": "June - July", "harvest": "Nov - Feb"},
}

# ---- Simple rule-based Farming Chatbot (no external API needed / FREE) ----
FAQ = [
    (["hello", "hi", "hey", "namaste", "namaskar"],
     "Namaste! 🌱 Main aapka Farming Assistant hoon. Crop selection, soil, fertilizer ya irrigation - kuch bhi puchiye!"),
    (["nitrogen", "n level", "urea"],
     "Nitrogen (N) paudhon ke green growth aur leaf development ke liye zaroori hai. Kami hone par patte peele pad jaate hain. Urea ya compost se N badha sakte hain."),
    (["phosphorus", "phosphorous", "p level", "dap"],
     "Phosphorus (P) jadon (roots) aur flowering ke liye important hai. DAP (Di-Ammonium Phosphate) ek accha source hai."),
    (["potassium", "k level", "mop"],
     "Potassium (K) plant ki overall strength, disease resistance aur fruit quality improve karta hai. MOP (Muriate of Potash) common source hai."),
    (["ph", "soil ph", "acidic", "alkaline"],
     "Zyaadatar crops ke liye ideal soil pH 6.0-7.5 hota hai. 6 se kam ho to lime dalein, 7.5 se zyada ho to organic matter/sulphur use karein."),
    (["rainfall", "water", "irrigation", "paani"],
     "Har crop ki alag water requirement hoti hai - jaise rice ko zyada paani chahiye (200mm+), jabki chickpea/lentil kam paani mein bhi ho jaate hain. App ke rainfall field mein apne region ka avg. rainfall daalein."),
    (["temperature", "garmi", "thand", "climate"],
     "Temperature crop selection ka bada factor hai. Zyaadatar crops 20-30°C range pasand karte hain, lekin coffee/apple jaise crops ko cooler climate chahiye."),
    (["fertilizer", "khaad", "manure", "compost"],
     "Balanced fertilizer (N-P-K) soil test ke hisaab se dalein. Organic compost/vermicompost soil health ke liye best rehta hai long-term mein."),
    (["pest", "keeda", "disease", "bimari", "fungus"],
     "Pest/disease control ke liye crop rotation follow karein, neem-based organic pesticide try karein, aur affected area ko turant isolate karein."),
    (["season", "kharif", "rabi", "zaid"],
     "Kharif season (June-Oct): rice, maize, cotton. Rabi season (Oct-Mar): wheat, chickpea, lentil. Zaid season (Mar-June): watermelon, muskmelon."),
    (["thanks", "thank you", "shukriya", "dhanyavad"],
     "Aapka swagat hai! 🙏 Kheti mein shubhkamnayein!"),
    (["rice", "chawal", "dhan"],
     "Rice ko zyada paani (200mm+ rainfall) aur garam-nam climate (20-30°C, high humidity) chahiye. Clayey soil best rehti hai, pH 5.5-6.5."),
    (["wheat", "gehu"],
     "Wheat Rabi crop hai (Oct-Nov sowing), cool climate (15-20°C) pasand karta hai, loamy soil aur pH 6-7.5 ideal hai."),
    (["cotton", "kapas"],
     "Cotton ko warm climate (21-30°C), kam se kam 6 mahine frost-free period, aur black/alluvial soil chahiye."),
    (["sugarcane", "ganna"],
     "Sugarcane ko garam-nam climate, deep loamy soil, aur bahut zyada paani chahiye (annual rainfall 750mm+)."),
    (["organic", "natural farming"],
     "Organic farming mein chemical fertilizer/pesticide ki jagah compost, vermicompost, neem-based pest control aur crop rotation use hota hai. Soil health long-term better rehti hai."),
    (["crop rotation"],
     "Crop rotation matlab har season/year alag family ke crops lagana (jaise legume ke baad cereal) - isse soil nutrients balance rehte hain aur pest cycle todta hai."),
    (["yield", "upaj", "production badhana"],
     "Yield badhane ke liye: soil testing karke sahi fertilizer dosage, timely irrigation, quality seeds, aur pest/disease ka time par control zaroori hai."),
    (["mixed cropping", "intercropping"],
     "Intercropping mein ek hi field mein do complementary crops saath lagaye jaate hain (jaise maize + legume) - land aur nutrients ka better use hota hai."),
    (["organic manure", "vermicompost", "fym"],
     "Vermicompost aur FYM (Farm Yard Manure) soil ki water-holding capacity aur microbial activity badhate hain, long-term soil health ke liye best hain."),
    (["drip irrigation", "sprinkler"],
     "Drip irrigation paani ki bahut bachat karta hai (jadon tak directly paani), especially fruits/vegetables aur kam-paani wale crops ke liye best."),
    (["how does this app work", "yeh app kaise kaam karta", "model kaise kaam karta"],
     "Yeh app N, P, K, temperature, humidity, pH, rainfall values lekar ek Random Forest ML model se best-fit crop predict karta hai - 2200+ real field samples par trained hai."),
]
FAQ_FALLBACK = ("Maaf kijiye, is sawaal ka exact jawab mere paas nahi hai (main ek simple rule-based bot hoon, "
                "free rakhne ke liye). Aap N, P, K, soil pH, rainfall, temperature, fertilizer, pest control, "
                "crop rotation, ya kisi specific crop (rice/wheat/cotton) ke baare mein puch sakte hain 🌾")


def get_chatbot_reply(message: str) -> str:
    msg = message.lower().strip()
    for keywords, reply in FAQ:
        if any(kw in msg for kw in keywords):
            return reply
    return FAQ_FALLBACK


# ---- SQLite for analytics (recommendation history/counts) ---------------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop TEXT NOT NULL,
                n REAL, p REAL, k REAL, temperature REAL,
                humidity REAL, ph REAL, rainfall REAL,
                created_at TEXT NOT NULL
            )
        """)
        db.commit()
    return db


@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


# ---- Routes: pages --------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", crops=sorted(CROP_DICT.keys()))


# ---- API: prediction -------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        values = [float(data[f]) for f in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input. Please provide N, P, K, temperature, humidity, ph, rainfall as numbers."}), 400

    features = pd.DataFrame([values], columns=FEATURE_ORDER)
    features_mm = minmax_scaler.transform(features)
    features_sc = standard_scaler.transform(features_mm)

    pred_num = int(model.predict(features_sc)[0])
    proba = model.predict_proba(features_sc)[0]
    confidence = float(np.max(proba)) * 100

    crop_name = REVERSE_CROP_DICT.get(pred_num, "unknown")

    # top-3 alternatives
    top_idx = np.argsort(proba)[::-1][:3]
    classes = model.classes_
    top3 = [
        {"crop": REVERSE_CROP_DICT.get(int(classes[i]), "unknown"),
         "confidence": round(float(proba[i]) * 100, 2)}
        for i in top_idx
    ]

    # save to history db for analytics
    db = get_db()
    db.execute(
        "INSERT INTO predictions (crop, n, p, k, temperature, humidity, ph, rainfall, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (crop_name, *values, datetime.now().isoformat()),
    )
    db.commit()

    return jsonify({
        "crop": crop_name,
        "emoji": CROP_EMOJI.get(crop_name, "🌱"),
        "confidence": round(confidence, 2),
        "top3": top3,
        "calendar": CROP_CALENDAR.get(crop_name, {}),
        "inputs": dict(zip(FEATURE_ORDER, values)),
    })


# ---- API: analytics dashboard ---------------------------------------------
@app.route("/api/analytics")
def api_analytics():
    db = get_db()
    rows = db.execute(
        "SELECT crop, COUNT(*) as cnt FROM predictions GROUP BY crop ORDER BY cnt DESC"
    ).fetchall()
    total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    recent = db.execute(
        "SELECT crop, n, p, k, temperature, humidity, ph, rainfall, created_at "
        "FROM predictions ORDER BY id DESC LIMIT 10"
    ).fetchall()

    return jsonify({
        "total_predictions": total,
        "by_crop": [{"crop": r[0], "count": r[1]} for r in rows],
        "recent": [
            {"crop": r[0], "N": r[1], "P": r[2], "K": r[3], "temperature": r[4],
             "humidity": r[5], "ph": r[6], "rainfall": r[7], "created_at": r[8]}
            for r in recent
        ],
    })


# ---- API: chatbot -----------------------------------------------------------
@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    data = request.get_json(force=True)
    message = data.get("message", "")
    if not message.strip():
        return jsonify({"reply": "Kuch to poochiye! 😊"})
    return jsonify({"reply": get_chatbot_reply(message)})


# ---- API: weather (proxy so API key stays server-side) ---------------------
@app.route("/api/weather")
def api_weather():
    city = request.args.get("city", "").strip()
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not OWM_API_KEY:
        return jsonify({"error": "Weather API key not configured. Set OWM_API_KEY environment variable "
                                  "(free key from https://openweathermap.org/api)."}), 501

    params = {"appid": OWM_API_KEY, "units": "metric"}
    if lat and lon:
        params["lat"] = lat
        params["lon"] = lon
    elif city:
        params["q"] = city
    else:
        return jsonify({"error": "Provide city or lat/lon"}), 400

    try:
        r = requests.get(OWM_BASE_URL, params=params, timeout=8)
        r.raise_for_status()
        w = r.json()
    except requests.RequestException as e:
        return jsonify({"error": f"Weather service unavailable: {e}"}), 502

    return jsonify({
        "city": w.get("name"),
        "temperature": w["main"]["temp"],
        "humidity": w["main"]["humidity"],
        "description": w["weather"][0]["description"],
        "icon": w["weather"][0]["icon"],
        "wind_speed": w["wind"]["speed"],
    })


# ---- API: crop info / calendar ---------------------------------------------
@app.route("/api/crop-info/<crop_name>")
def api_crop_info(crop_name):
    crop_name = crop_name.lower()
    if crop_name not in CROP_STATS:
        return jsonify({"error": "Unknown crop"}), 404
    return jsonify({
        "crop": crop_name,
        "emoji": CROP_EMOJI.get(crop_name, "🌱"),
        "ideal_ranges": CROP_STATS[crop_name],
        "calendar": CROP_CALENDAR.get(crop_name, {}),
    })


@app.route("/api/crop-calendar")
def api_crop_calendar():
    return jsonify({
        crop: {**CROP_CALENDAR[crop], "emoji": CROP_EMOJI.get(crop, "🌱")}
        for crop in CROP_CALENDAR
    })


# ---- API: PDF report --------------------------------------------------------
@app.route("/api/report", methods=["POST"])
def api_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    data = request.get_json(force=True)
    crop = data.get("crop", "N/A")
    confidence = data.get("confidence", "")
    inputs = data.get("inputs", {})
    calendar = data.get("calendar", {})

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=colors.HexColor("#2e7d32"))
    h2 = ParagraphStyle("H2Green", parent=styles["Heading2"], textColor=colors.HexColor("#2e7d32"))

    elements = [
        Paragraph("Crop Recommendation Report", title_style),
        Spacer(1, 6),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph(f"Recommended Crop: {crop.title()}", h2),
        Paragraph(f"Model Confidence: {confidence}%", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Input Soil & Climate Parameters", h2),
    ]

    table_data = [["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()]
    t = Table(table_data, colWidths=[80 * mm, 80 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f8f2")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    if calendar:
        elements.append(Paragraph("Crop Calendar", h2))
        cal_data = [["Season", "Sowing Time", "Harvest Time"],
                    [calendar.get("season", "-"), calendar.get("sow", "-"), calendar.get("harvest", "-")]]
        ct = Table(cal_data, colWidths=[53 * mm, 53 * mm, 53 * mm])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#558b2f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        elements.append(ct)

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Generated by Crop Recommendation System — Krishna Sharma", styles["Italic"]))

    doc.build(elements)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                      download_name=f"crop_report_{crop}.pdf")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
