import os
import json
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")

# Load model & feature names at startup
with open(FEATURES_PATH, encoding='utf-8') as f:
    FEATURE_NAMES = json.load(f)

# Label bahasa awam (untuk UI) — kode variabel tetap dipakai di name/id
FEATURE_LABELS = {
    "Present_Tmax": "Suhu Maksimum Hari Ini",
    "Present_Tmin": "Suhu Minimum Hari Ini",
    "LDAPS_RHmin": "Kelembapan Minimum (RHmin)",
    "LDAPS_RHmax": "Kelembapan Maksimum (RHmax)",
    "LDAPS_Tmax_lapse": "Laju Penurunan Suhu Maks (LDAPS)",
    "LDAPS_Tmin_lapse": "Laju Penurunan Suhu Min (LDAPS)",
    "LDAPS_WS": "Kecepatan Angin (WS)",
    "LDAPS_LH": "Panas Laten (LH)",
    "LDAPS_CC1": "Tutupan Awan 1 (CC1)",
    "LDAPS_CC2": "Tutupan Awan 2 (CC2)",
    "LDAPS_CC3": "Tutupan Awan 3 (CC3)",
    "LDAPS_CC4": "Tutupan Awan 4 (CC4)",
    "LDAPS_PPT1": "Curah Hujan 1 (PPT1)",
    "LDAPS_PPT2": "Curah Hujan 2 (PPT2)",
    "LDAPS_PPT3": "Curah Hujan 3 (PPT3)",
    "LDAPS_PPT4": "Curah Hujan 4 (PPT4)",
    "lat": "Lintang (Latitude)",
    "lon": "Bujur (Longitude)",
    "DEM": "Ketinggian Permukaan (DEM)",
    "Slope": "Kemiringan Lereng",
    "Solar radiation": "Radiasi Matahari",
}

MODEL = joblib.load(MODEL_PATH)
print(f"[i] Model loaded: {MODEL_PATH} | features: {len(FEATURE_NAMES)}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_names=FEATURE_NAMES, labels=FEATURE_LABELS)

@app.route("/predict", methods=["POST"])
def predict():
    # Ambil 21 fitur dari form
    data = {}
    for fn in FEATURE_NAMES:
        val = request.form.get(fn)
        if val is None or val.strip() == "":
            return render_template(
                "result.html",
                error=f"Field '{fn}' kosong. Lengkapi semua input atau gunakan Import CSV.",
                feature_names=FEATURE_NAMES,
                labels=FEATURE_LABELS,
            )
        try:
            data[fn] = float(val)
        except ValueError:
            return render_template(
                "result.html",
                error=f"Field '{fn}' harus angka numerik (contoh: 29.5).",
                feature_names=FEATURE_NAMES,
                labels=FEATURE_LABELS,
            )

    # Buat DataFrame 1 baris dengan urutan fitur yang benar
    df_input = pd.DataFrame([data])[FEATURE_NAMES]
    pred = MODEL.predict(df_input)[0]

    return render_template(
        "result.html",
        prediction=round(pred, 2),
        input_data=data,
        feature_names=FEATURE_NAMES,
        labels=FEATURE_LABELS,
        r2_score=0.8942,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)