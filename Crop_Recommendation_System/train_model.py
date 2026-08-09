"""
Crop Recommendation - Model Training Script
Reproduces the logic from Krishna's notebook (RandomForestClassifier, ~99% acc)
Fix vs original notebook: the scaler is now saved and applied at inference time
too (original notebook fit a scaler on x_train but the recommendation()
function fed raw unscaled values into rfc.predict(), which is inconsistent
with how the model was trained).
"""
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/Crop_recommendation.csv"

crop = pd.read_csv(DATA_PATH)

crop_dict = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

crop['crop_num'] = crop['label'].map(crop_dict)

x = crop.drop(['label', 'crop_num'], axis=1)
y = crop['crop_num']

FEATURE_ORDER = list(x.columns)  # ['N','P','K','temperature','humidity','ph','rainfall']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mx = MinMaxScaler()
x_train_mm = mx.fit_transform(x_train)
x_test_mm = mx.transform(x_test)

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train_mm)
x_test_sc = sc.transform(x_test_mm)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(x_train_sc, y_train)

pred = rfc.predict(x_test_sc)
acc = accuracy_score(y_test, pred)
print(f"Random Forest test accuracy: {acc:.4f}")
print(classification_report(y_test, pred, target_names=[reverse_crop_dict[i] for i in sorted(reverse_crop_dict)]))

# Also compute per-crop stats (min/max/avg of each feature) -> used for the
# frontend's "Crop Calendar" / ideal-condition reference and Chart.js radar.
crop_stats = {}
for label, cnum in crop_dict.items():
    sub = crop[crop['crop_num'] == cnum][FEATURE_ORDER]
    crop_stats[label] = {
        col: {
            "min": round(float(sub[col].min()), 2),
            "max": round(float(sub[col].max()), 2),
            "avg": round(float(sub[col].mean()), 2),
        } for col in FEATURE_ORDER
    }

with open("model/crop_recommendation_model.pkl", "wb") as f:
    pickle.dump(rfc, f)
with open("model/minmax_scaler.pkl", "wb") as f:
    pickle.dump(mx, f)
with open("model/standard_scaler.pkl", "wb") as f:
    pickle.dump(sc, f)
with open("model/crop_dict.json", "w") as f:
    json.dump({"crop_dict": crop_dict, "reverse_crop_dict": reverse_crop_dict,
               "feature_order": FEATURE_ORDER}, f, indent=2)
with open("model/crop_stats.json", "w") as f:
    json.dump(crop_stats, f, indent=2)

print("Saved model, scalers, crop_dict.json, crop_stats.json to model/")
