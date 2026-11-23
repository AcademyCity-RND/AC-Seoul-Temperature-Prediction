import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Path ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_URL = (
    "https://raw.githubusercontent.com/matthewpriantara/"
    "Data_Mining_Practice/main/uts/Bias_correction_ucl.csv"
)
DATA_PATH = os.path.join(BASE_DIR, "Bias_correction_ucl.csv")
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.json")

# ── 1. Muat dataset ──────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    print("[i] Mengunduh dataset dari GitHub ...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_PATH, index=False)
else:
    print(f"[i] Membaca dataset lokal: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

print(f"Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")

# ── 2. Preprocessing ─────────────────────────────────────────────
df_clean = df.dropna().copy()
fitur_drop = ['station', 'Date', 'Next_Tmax', 'Next_Tmin']
X = df_clean.drop(columns=fitur_drop)
y = df_clean['Next_Tmax']

feature_names = list(X.columns)
print(f"Fitur masukan: {len(feature_names)} -> {feature_names}")
print(f"Target: Next_Tmax ({len(y)} sampel)")

# ── 3. Train-Test Split (80:20) ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} sampel | Test: {X_test.shape[0]} sampel")

# ── 4. Scaling (hanya untuk Linear Regression) ──────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 5. Training ──────────────────────────────────────────────────
results = {}

# 5.1 Linear Regression (scaled)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = {
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'RMSE': mean_squared_error(y_test, y_pred_lr) ** 0.5,
    'R2': r2_score(y_test, y_pred_lr),
}
print(f"Linear Regression  : MAE={results['Linear Regression']['MAE']:.4f} "
      f"RMSE={results['Linear Regression']['RMSE']:.4f} R2={results['Linear Regression']['R2']:.4f}")

# 5.2 Decision Tree Regressor (unscaled)
dt = DecisionTreeRegressor(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'MAE': mean_absolute_error(y_test, y_pred_dt),
    'RMSE': mean_squared_error(y_test, y_pred_dt) ** 0.5,
    'R2': r2_score(y_test, y_pred_dt),
}
print(f"Decision Tree      : MAE={results['Decision Tree']['MAE']:.4f} "
      f"RMSE={results['Decision Tree']['RMSE']:.4f} R2={results['Decision Tree']['R2']:.4f}")

# 5.3 Random Forest Regressor (unscaled) — model terbaik
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'RMSE': mean_squared_error(y_test, y_pred_rf) ** 0.5,
    'R2': r2_score(y_test, y_pred_rf),
}
print(f"Random Forest      : MAE={results['Random Forest']['MAE']:.4f} "
      f"RMSE={results['Random Forest']['RMSE']:.4f} R2={results['Random Forest']['R2']:.4f}")

# ── 6. Simpan model terbaik + scaler + nama fitur ────────────────
joblib.dump(rf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=2)

print(f"\nModel disimpan: {MODEL_PATH}")
print(f"Scaler disimpan : {SCALER_PATH}")
print(f"Feature names   : {FEATURES_PATH}")

# ── 7. Feature Importance (Random Forest) ────────────────────────
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\n=== Top 5 Feature Importance ===")
for col, val in feat_imp.head(5).items():
    print(f"{col:18s}: {val*100:.2f}%")

print("\nSelesai! Model siap dipakai di app_weather.py")
