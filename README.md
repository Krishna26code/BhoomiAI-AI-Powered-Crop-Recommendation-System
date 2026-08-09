# 🌾 BhoomiAI — Smart Crop Recommendation System

BhoomiAI is a machine-learning powered web app that recommends the most suitable crop for a farmer to grow, based on soil nutrients (N-P-K), soil pH, and local climate conditions (temperature, humidity, rainfall).

Built with **Flask** + **scikit-learn**, trained on 2,200+ real agricultural samples covering 22 different crops grown across India.

---

## 🎯 Live Demo

Run locally — see Setup below. (Deploy for free on Render / Railway / PythonAnywhere and drop your link here.)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌱 Crop Recommendation | Enter N, P, K, temperature, humidity, pH, rainfall → get the best-fit crop with confidence score + top-3 alternatives |
| 🤖 Farming Chatbot | Rule-based Q&A assistant for soil nutrients, fertilizers, pest control, crop rotation, irrigation, and more |
| 📊 Analytics Dashboard | Live chart of how often each crop has been recommended, powered by Chart.js |
| 🌦️ Live Weather Widget | Real-time weather via OpenWeatherMap — auto-fill temperature & humidity into the prediction form |
| 📋 Prediction History | Last 10 predictions saved locally in your browser (localStorage) |
| 🌍 Hindi / English Toggle | Full bilingual interface |
| 📱 Installable PWA | Add to home screen on mobile, works with an offline app shell |
| 📄 PDF Report | One-click downloadable recommendation report |
| 🗺️ Crop Calendar | Season-wise sowing & harvest windows for all 22 crops |

---

## 🧠 Model & Accuracy

Ten classification algorithms were trained and compared on the dataset before picking the final model:

| Algorithm | Accuracy |
|---|---|
| Naive Bayes | 99.5% |
| Random Forest (used in production) | 99.3% |
| Bagging Classifier | 99.1% |
| Decision Tree | 98.4% |
| Gradient Boosting | 98.2% |
| Support Vector Machine | 96.8% |
| K-Nearest Neighbors | 96.6% |
| Logistic Regression | 96.4% |
| Extra Trees | 88.9% |
| AdaBoost | 14.5% |

Random Forest was chosen for production over the marginally higher-scoring Naive Bayes because ensemble tree models tend to generalize better and are more robust to noisy/real-world input than a single-pass probabilistic model — a safer choice for a tool farmers would actually rely on.

Pipeline: MinMaxScaler → StandardScaler → RandomForestClassifier (random_state=42 for reproducibility). The scaler is fit during training and re-applied identically at inference time, so predictions stay consistent with what the model was trained on.

---

## 🌾 Crops Covered (22)

Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee.

---

## 🛠️ Tech Stack

- Backend: Flask, scikit-learn, pandas, numpy, SQLite
- Frontend: HTML/CSS/vanilla JS, Chart.js
- PDF generation: ReportLab
- Weather: OpenWeatherMap API (free tier)
- PWA: Web App Manifest + Service Worker

---

## 📁 Project Structure

BhoomiAI/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── Crop_recommendation.csv
├── model/
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

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| /api/predict | POST | Get crop recommendation from soil/climate values |
| /api/analytics | GET | Recommendation counts for the dashboard |
| /api/chatbot | POST | Farming Q&A chatbot reply |
| /api/weather | GET | Live weather (proxied, key stays server-side) |
| /api/report | POST | Generate PDF recommendation report |
| /api/crop-info/<crop> | GET | Ideal condition ranges for a crop |
| /api/crop-calendar | GET | Season-wise calendar for all crops |

---

## 🙌 Credits

Built by Krishna Sharma. Dataset: standard Crop Recommendation dataset (N-P-K, temperature, humidity, pH, rainfall → 22 crop labels, 2,200 samples).

---

## 📄 License

Free to use for learning and personal projects.
