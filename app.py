from pathlib import Path
from datetime import datetime
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DB_PATH = ROOT / "agriculture.db"

app = Flask(__name__)
app.secret_key = "change-this-development-secret-key"

production_model = None
categories = []
units = {}
metrics = {}
detector_file = MODEL_DIR / "crop_weed_best.pt"
model_file = MODEL_DIR / "production_model.pkl"
if model_file.exists():
    production_model = joblib.load(model_file)
    categories_file = MODEL_DIR / "categories.pkl"
    units_file = MODEL_DIR / "units.pkl"
    metrics_file = MODEL_DIR / "metrics.pkl"
    categories = joblib.load(categories_file) if categories_file.exists() else []
    units = joblib.load(units_file) if units_file.exists() else {}
    metrics = joblib.load(metrics_file) if metrics_file.exists() else {}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS production_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                year INTEGER NOT NULL,
                predicted_production REAL NOT NULL,
                unit TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weed_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                result_filename TEXT,
                detections_json TEXT,
                created_at TEXT NOT NULL
            )
        """)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html", categories=categories, metrics=metrics,
                           model_ready=production_model is not None,
                           detector_ready=detector_file.exists())

@app.route("/predict-production", methods=["POST"])
def predict_production():
    if production_model is None:
        flash("Production model is not ready. Run preprocess.py and train_model.py first.", "error")
        return redirect(url_for("home"))
    try:
        category = request.form.get("category", "").strip()
        year = int(request.form.get("year", ""))
        if category not in categories:
            raise ValueError("Choose a category from the list.")
        if not 1993 <= year <= 2035:
            raise ValueError("Enter a year from 1993 to 2035.")
        row = pd.DataFrame([{"Particulars": category, "Year": year}])
        value = float(production_model.predict(row)[0])
        unit = units.get(category, "source unit not recorded")
        # Store the estimate
        with get_db() as conn:
            conn.execute(
                "INSERT INTO production_predictions(category,year,predicted_production,unit,created_at) VALUES(?,?,?,?,?)",
                (category, year, value, unit, datetime.now().isoformat(timespec="seconds"))
            )
        return render_template("production_result.html", category=category, year=year,
                               value=round(value, 4), unit=unit, future=year > 2014)
    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("home"))

@app.route("/weed-detection", methods=["GET", "POST"])
def weed_detection():
    result_image = None
    detections = []
    error = None
    if request.method == "POST":
        upload = request.files.get("image")
        if not upload or not upload.filename:
            error = "Choose an image file first."
        elif not allowed_file(upload.filename):
            error = "Supported formats: PNG, JPG, JPEG, WEBP."
        else:
            safe_name = secure_filename(upload.filename)
            stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            input_name = f"{stamp}_{safe_name}"
            input_path = UPLOAD_DIR / input_name
            upload.save(input_path)

            detector_path = MODEL_DIR / "crop_weed_best.pt"
            if not detector_path.exists():
                error = (
                    "The crop/weed model has not been trained yet. Add your YOLO-labelled dataset "
                    "under dataset/weed_detection, configure data.yaml, and run train_yolo.py. "
                    "The uploaded image is saved locally but has not been classified."
                )
            else:
                try:
                    from ultralytics import YOLO
                    detector = YOLO(str(detector_path))
                    prediction = detector.predict(source=str(input_path), imgsz=512, conf=0.25, verbose=False)[0]
                    names = prediction.names
                    if prediction.boxes is not None:
                        for box in prediction.boxes:
                            cls_id = int(box.cls.item())
                            detections.append({
                                "label": names.get(cls_id, str(cls_id)),
                                "confidence": round(float(box.conf.item()) * 100, 1),
                                "x1": round(float(box.xyxy[0][0].item()), 1),
                                "y1": round(float(box.xyxy[0][1].item()), 1),
                                "x2": round(float(box.xyxy[0][2].item()), 1),
                                "y2": round(float(box.xyxy[0][3].item()), 1)
                            })
                    result_name = f"detected_{input_name.rsplit('.', 1)[0]}.jpg"
                    result_path = UPLOAD_DIR / result_name
                    prediction.save(filename=str(result_path))
                    result_image = f"uploads/{result_name}"
                    import json
                    with get_db() as conn:
                        conn.execute(
                            "INSERT INTO weed_detections(original_filename,result_filename,detections_json,created_at) VALUES(?,?,?,?)",
                            (safe_name, result_name, json.dumps(detections), datetime.now().isoformat(timespec="seconds"))
                        )
                except Exception as exc:
                    error = f"Detection failed: {exc}"
    return render_template("weed_detection.html", result_image=result_image,
                           detections=detections, error=error,
                           detector_ready=(MODEL_DIR / "crop_weed_best.pt").exists())

@app.route("/history")
def history():
    with get_db() as conn:
        production = conn.execute(
            "SELECT * FROM production_predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
        detections = conn.execute(
            "SELECT * FROM weed_detections ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return render_template("history.html", production=production, detections=detections)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
