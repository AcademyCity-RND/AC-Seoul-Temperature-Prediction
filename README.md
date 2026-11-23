# AC-Seoul-Temperature-Prediction

Dashboard Flask untuk memprediksi **suhu maksimum esok hari di Seoul, Korea Selatan** (`Next_Tmax`) menggunakan model **Random Forest Regressor** yang dilatih pada dataset **UCI Bias Correction of Numerical Prediction Model Temperature Forecast** (ID 514).

---

## 📖 Deskripsi Proyek

Tugas individu Praktikum Penambangan Data — memprediksi `Next_Tmax` (suhu maksimum esok hari) di Seoul, Korea Selatan.

**Paper Rujukan:** Cho et al. (2020), *Earth and Space Science*, AGU — DOI: [10.1029/2019EA000740](https://doi.org/10.1029/2019EA000740)

---

## 📊 Dataset (Seoul, Korea Selatan)

- **Sumber:** [UCI Machine Learning Repository — Bias Correction Temperature Forecast](https://archive.ics.uci.edu/dataset/514/bias+correction+of+numerical+prediction+model+temperature+forecast)
- **Mirror GitHub (raw CSV):** `https://raw.githubusercontent.com/matthewpriantara/Data_Mining_Practice/main/uts/Bias_correction_ucl.csv`
- **Ukuran:** 7.752 baris × 25 kolom
- **Wilayah studi:** Seoul, Korea Selatan (koordinat kota ≈ 37.5665° N, 126.9780° E)
- **Target:** `Next_Tmax` — suhu maksimum esok hari (°C) di Seoul
- **Fitur masukan (21):**
  `Present_Tmax`, `Present_Tmin`, `LDAPS_RHmin`, `LDAPS_RHmax`, `LDAPS_Tmax_lapse`, `LDAPS_Tmin_lapse`, `LDAPS_WS`, `LDAPS_LH`, `LDAPS_CC1`, `LDAPS_CC2`, `LDAPS_CC3`, `LDAPS_CC4`, `LDAPS_PPT1`, `LDAPS_PPT2`, `LDAPS_PPT3`, `LDAPS_PPT4`, `lat`, `lon`, `DEM`, `Slope`, `Solar radiation`

---

## ⚙️ Instalasi

```bash
# 1. Clone repo
git clone <repo-url>
cd AC-Seoul-Temperature-Prediction

# 2. Aktifkan virtual environment (sudah disediakan di root PPD10)
..\.venv\Scripts\activate   # Windows
# source ../.venv/bin/activate  # Linux/Mac

# 3. Install dependencies (jika belum)
pip install flask pandas scikit-learn joblib
```

> **Catatan:** Virtual environment di `PPD10/.venv` sudah terinstall Flask 3.1.3, pandas 3.1.1, scikit-learn 1.9.1, joblib.

---

## 🏋️ Training Model

```bash
python train_weather.py
```

Output:
- `rf_model.joblib` — Random Forest Regressor (n_estimators=100, random_state=42)
- `scaler.joblib` — StandardScaler (fit pada X_train, untuk konsistensi)
- `feature_names.json` — Daftar 21 nama fitur (urutan pasti)
- `Bias_correction_ucl.csv` — Dataset lokal (auto-download saat pertama kali)

**Hasil evaluasi (test set 20%, random_state=42):**

| Model | MAE (°C) | RMSE (°C) | R² Score |
|-------|----------|-----------|----------|
| Linear Regression (scaled) | 1.1189 | 1.4704 | 0.7704 |
| Decision Tree Regressor | 1.0837 | 1.4415 | 0.7794 |
| **Random Forest Regressor** ⭐ | **0.7398** | **0.9984** | **0.8942** |

Baseline LDAPS (`LDAPS_Tmax_lapse` saja): R² = 0.6154
**Model terbaik melampaui target paper Cho et al. (2020): R² > 0.75.**

**Top 5 Feature Importance (Random Forest):**
1. `LDAPS_Tmax_lapse` — 69.52%
2. `Present_Tmax` — 5.37%
3. `LDAPS_CC1` — 3.06%
4. `LDAPS_CC3` — 2.42%
5. `LDAPS_CC4` — 2.29%

---

## 🚀 Menjalankan Dashboard Flask

```bash
python app_weather.py
```

Server berjalan di: **http://127.0.0.1:5000**

### Fitur Dashboard
- **Form input 21 fitur** (responsive grid, validasi numerik, label bahasa awam + kode)
- **Import CSV** — upload file dengan header 21 kolom (urutan bebas), form terisi otomatis
- **Download template CSV** — 1 baris contoh data
- **Hasil prediksi** — suhu dalam °C, info model (R² = 0.8942), ringkasan input 2 tabel (11 + 10) dalam 1 layar

---

## 📁 Struktur Proyek

```
AC-Seoul-Temperature-Prediction/
├── train_weather.py          # Training pipeline + simpan model
├── app_weather.py            # Flask app (GET /, POST /predict)
├── Bias_correction_ucl.csv   # Dataset (1.65 MB, auto-download)
├── rf_model.joblib           # Model Random Forest terlatih
├── scaler.joblib             # StandardScaler (fit X_train)
├── feature_names.json        # 21 nama fitur (urutan pasti)
├── static/                   # SB Admin 2 (Bootstrap, Chart.js, FontAwesome, dll)
└── templates/
    ├── index.html            # Form input + Import CSV dropdown
    └── result.html           # Hasil prediksi + ringkasan input (2 tabel)
```

---

## 🔌 API Endpoint

### `GET /`
Mengembalikan halaman form input (HTML).

### `POST /predict`
Menerima form-data 21 field numerik, mengembalikan halaman hasil prediksi (HTML).

**Contoh request (curl):**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d "Present_Tmax=28.7" \
  -d "Present_Tmin=21.4" \
  -d "LDAPS_RHmin=58.255688" \
  -d "LDAPS_RHmax=91.116364" \
  -d "LDAPS_Tmax_lapse=28.074101" \
  -d "LDAPS_Tmin_lapse=23.006936" \
  -d "LDAPS_WS=6.818887" \
  -d "LDAPS_LH=69.451805" \
  -d "LDAPS_CC1=0.0" \
  -d "LDAPS_CC2=0.0" \
  -d "LDAPS_CC3=0.0" \
  -d "LDAPS_CC4=0.0" \
  -d "LDAPS_PPT1=0.0" \
  -d "LDAPS_PPT2=0.0" \
  -d "LDAPS_PPT3=0.0" \
  -d "LDAPS_PPT4=0.0" \
  -d "lat=37.6046" \
  -d "lon=126.991" \
  -d "DEM=212.335" \
  -d "Slope=2.785" \
  -d "Solar radiation=5992.895996"
```

**Response:** HTML halaman `result.html` dengan prediksi (contoh: **29.65 °C**).

---

## 🧪 Verifikasi Cepat (Test Client)

```python
import json
import app_weather

c = app_weather.app.test_client()
sample = json.load(open("sample_input.json"))  # 21 field
r = c.post("/predict", data={k: str(v) for k, v in sample.items()})
assert r.status_code == 200
assert "29.65" in r.data.decode()  # prediksi baris pertama notebook
```

---

## 📝 Lisensi & Referensi

- Dataset: UCI ML Repository (CC BY 4.0)
- Paper: Cho et al. (2020), *Earth and Space Science*, AGU
- Template UI: [Start Bootstrap SB Admin 2](https://startbootstrap.com/theme/sb-admin-2) (MIT License)
- Kode: Praktikum Penambangan Data — Institut Teknologi Bandung

---

## 🤝 Kontribusi

Pull request welcomed untuk perbaikan dokumentasi, visualisasi tambahan, atau eksperimen model lain (XGBoost, LightGBM, dll).

