import re
import json
import app_weather

c = app_weather.app.test_client()

print("=" * 50)
print("TEST 1: GET / (halaman form)")
r = c.get("/")
print("  status:", r.status_code, "| bytes:", len(r.data))
assert r.status_code == 200

print("TEST 2: POST /predict (data valid)")
sample = json.load(open("sample_input.json"))
r = c.post("/predict", data={k: str(v) for k, v in sample.items()})
body = r.data.decode()
pred = re.search(r'result-value">([\-0-9.]+)', body)
print("  status:", r.status_code, "| prediksi:", pred.group(1) if pred else "NOT FOUND")
assert r.status_code == 200 and pred

print("TEST 3: POST /predict (field kosong)")
bad = dict(sample); bad["DEM"] = ""
r = c.post("/predict", data={k: str(v) for k, v in bad.items()})
print("  status:", r.status_code, "| error ditampilkan:", "kosong" in r.data.decode())
assert "kosong" in r.data.decode()

print("TEST 4: POST /predict (field bukan angka)")
bad = dict(sample); bad["Present_Tmax"] = "abc"
r = c.post("/predict", data={k: str(v) for k, v in bad.items()})
print("  status:", r.status_code, "| error ditampilkan:", "numerik" in r.data.decode())
assert "numerik" in r.data.decode()

print("=" * 50)
print("SEMUA TEST LULUS ✓")
