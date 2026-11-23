"""Modal deployment for AC-Seoul-Temperature-Prediction Flask dashboard.

Deploy:
  modal deploy deploy_modal.py

Login first if you haven't:
  modal setup   # opens browser for auth
"""

import modal

app = modal.App("ac-seoul-temperature-prediction")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flask==3.0.3", "joblib==1.4.2", "pandas==2.2.2", "scikit-learn==1.5.1", "numpy==1.26.4", "gunicorn==23.0.0")
    .add_local_file("app_weather.py", "/root/app_weather.py")
    .add_local_file("feature_names.json", "/root/feature_names.json")
    .add_local_file("rf_model.joblib", "/root/rf_model.joblib")
    .add_local_file("scaler.joblib", "/root/scaler.joblib")
    .add_local_dir("templates", "/root/templates")
    .add_local_dir("static", "/root/static")
)

@app.function(image=image)
@modal.wsgi_app()
def flask_app():
    import app_weather
    return app_weather.app
