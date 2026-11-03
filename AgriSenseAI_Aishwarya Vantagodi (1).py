
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "static" / "uploads"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "agriculture.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"

# --------------------------------------------------
# 2. LOAD PRODUCTION MODEL
# --------------------------------------------------

production_model = None
categories = []
units = {}
metrics = {}

model_file = MODEL_DIR / "production_model.pkl"
categories_file = MODEL_DIR / "categories.pkl"
units_file = MODEL_DIR / "units.pkl"
metrics_file = MODEL_DIR / "metrics.pkl"

try:
    if model_file.exists():
        production_model = joblib.load(model_file)

    if categories_file.exists():
        categories = joblib.load(categories_file)

    if units_file.exists():
        units = joblib.load(units_file)

    if metrics_file.exists():
        metrics = joblib.load(metrics_file)

except Exception as exc:
    print("Production model loading warning:", exc)

# YOLO weed detection model
detector_file = MODEL_DIR / "crop_weed_best.pt"


# --------------------------------------------------
# 3. DATABASE CONNECTION
# --------------------------------------------------

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


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_dashboard_data():
    data = {
        "total_scans": 0,
        "total_weeds": 0,
        "average_confidence": 0,
        "class_labels": [],
        "class_counts": [],
        "daily_labels": [],
        "daily_counts": [],
        "production_total": 0,
        "year_labels": [],
        "year_counts": [],
        "recent_detections": [],
        "recent_predictions": []
    }

    try:
        with get_db() as conn:
            detections = conn.execute("""
                SELECT *
                FROM weed_detections
                ORDER BY id DESC
            """).fetchall()

            predictions = conn.execute("""
                SELECT *
                FROM production_predictions
                ORDER BY id DESC
            """).fetchall()

        data["total_scans"] = len(detections)
        data["production_total"] = len(predictions)

        data["recent_detections"] = [
            dict(row) for row in detections[:5]
        ]

        data["recent_predictions"] = [
            dict(row) for row in predictions[:5]
        ]

        class_counts = {}
        confidence_values = []
        daily_counts = {}

        for row in detections:
            try:
                items = json.loads(row["detections_json"] or "[]")
                if not isinstance(items, list):
                    items = []
            except (TypeError, json.JSONDecodeError):
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue

                label = str(item.get("label", "Unknown"))

                class_counts[label] = (
                    class_counts.get(label, 0) + 1
                )

                if "weed" in label.lower():
                    data["total_weeds"] += 1

                try:
                    confidence_values.append(
                        float(item.get("confidence", 0))
                    )
                except (TypeError, ValueError):
                    pass

            day = (row["created_at"] or "")[:10]

            if day:
                daily_counts[day] = (
                    daily_counts.get(day, 0) + len(items)
                )

        if confidence_values:
            data["average_confidence"] = round(
                sum(confidence_values) / len(confidence_values), 1
            )

        data["class_labels"] = (
            list(class_counts.keys()) or ["No detections yet"]
        )

        data["class_counts"] = (
            list(class_counts.values()) or [1]
        )

        last_days = [
            (datetime.now().date() - timedelta(days=i)).isoformat()
            for i in range(6, -1, -1)
        ]

        data["daily_labels"] = [
            datetime.strptime(day, "%Y-%m-%d").strftime("%b %d")
            for day in last_days
        ]

        data["daily_counts"] = [
            daily_counts.get(day, 0) for day in last_days
        ]

        year_counts = {}

        for row in predictions:
            year = str(row["year"])
            year_counts[year] = year_counts.get(year, 0) + 1

        sorted_years = sorted(year_counts.keys())

        data["year_labels"] = sorted_years or ["No data"]
        data["year_counts"] = [
            year_counts[year] for year in sorted_years
        ] or [0]

    except sqlite3.Error as exc:
        print("Dashboard database warning:", exc)

    return data


# --------------------------------------------------
# 5. MAIN DASHBOARD
# --------------------------------------------------

@app.route("/")
def home():
    init_db()

    dashboard = get_dashboard_data()

    return render_template(
        "index.html",
        categories=categories,
        metrics=metrics,
        model_ready=production_model is not None,
        detector_ready=detector_file.exists(),
        dashboard=dashboard
    )


# --------------------------------------------------
# 6. CROP PRODUCTION PREDICTION
# --------------------------------------------------

@app.route("/predict-production", methods=["POST"])
def predict_production():

    if production_model is None:
        flash(
            "Production model is not ready. "
            "Check your model files in the models folder.",
            "error"
        )
        return redirect(url_for("home"))

    try:
        category = request.form.get("category", "").strip()
        year = int(request.form.get("year", ""))

        if category not in categories:
            raise ValueError("Please select a valid crop category.")

        if not 1993 <= year <= 2035:
            raise ValueError("Enter a year between 1993 and 2035.")

        # Keep the feature names consistent with model training.
        input_data = pd.DataFrame([{
            "Particulars": category,
            "Year": year
        }])

        prediction = float(
            production_model.predict(input_data)[0]
        )

        unit = units.get(category, "unit not recorded")

        with get_db() as conn:
            conn.execute("""
                INSERT INTO production_predictions
                (category, year, predicted_production, unit, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                category,
                year,
                prediction,
                unit,
                datetime.now().isoformat(timespec="seconds")
            ))

        return render_template(
            "production_result.html",
            category=category,
            year=year,
            value=round(prediction, 4),
            unit=unit,
            future=year > datetime.now().year
        )

    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("home"))

    except Exception as exc:
        app.logger.exception("Production prediction failed")
        flash(
            "Prediction failed. Check the model and input features.",
            "error"
        )
        return redirect(url_for("home"))


# --------------------------------------------------
# 7. WEED DETECTION USING YOLO
# --------------------------------------------------

@app.route("/weed-detection", methods=["GET", "POST"])
def weed_detection():

    result_image = None
    detections = []
    error = None

    if request.method == "POST":
        upload = request.files.get("image")

        if not upload or not upload.filename:
            error = "Please choose an image first."

        elif not allowed_file(upload.filename):
            error = "Upload a PNG, JPG, JPEG, or WEBP image."

        elif not detector_file.exists():
            error = (
                "YOLO model not found. Place crop_weed_best.pt "
                "inside the models folder."
            )

        else:
            safe_name = secure_filename(upload.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

            input_name = f"{timestamp}_{safe_name}"
            input_path = UPLOAD_DIR / input_name

            upload.save(input_path)

            try:
                from ultralytics import YOLO

                detector = YOLO(str(detector_file))

                results = detector.predict(
                    source=str(input_path),
                    imgsz=512,
                    conf=0.25,
                    verbose=False
                )

                prediction = results[0]
                names = prediction.names

                if prediction.boxes is not None:
                    for box in prediction.boxes:
                        class_id = int(box.cls.item())

                        if isinstance(names, dict):
                            label = names.get(
                                class_id, str(class_id)
                            )
                        else:
                            label = names[class_id]

                        detections.append({
                            "label": str(label),
                            "confidence": round(
                                float(box.conf.item()) * 100, 1
                            ),
                            "x1": round(
                                float(box.xyxy[0][0].item()), 1
                            ),
                            "y1": round(
                                float(box.xyxy[0][1].item()), 1
                            ),
                            "x2": round(
                                float(box.xyxy[0][2].item()), 1
                            ),
                            "y2": round(
                                float(box.xyxy[0][3].item()), 1
                            )
                        })

                result_name = (
                    f"detected_{input_name.rsplit('.', 1)[0]}.jpg"
                )

                result_path = UPLOAD_DIR / result_name
                prediction.save(filename=str(result_path))

                result_image = f"uploads/{result_name}"

                with get_db() as conn:
                    conn.execute("""
                        INSERT INTO weed_detections
                        (
                            original_filename,
                            result_filename,
                            detections_json,
                            created_at
                        )
                        VALUES (?, ?, ?, ?)
                    """, (
                        safe_name,
                        result_name,
                        json.dumps(detections),
                        datetime.now().isoformat(timespec="seconds")
                    ))

            except Exception as exc:
                app.logger.exception("Weed detection failed")
                error = f"Weed detection failed: {exc}"

    return render_template(
        "weed_detection.html",
        result_image=result_image,
        detections=detections,
        error=error,
        detector_ready=detector_file.exists()
    )


# --------------------------------------------------
# 8. HISTORY PAGE
# --------------------------------------------------

@app.route("/history")
def history():
    init_db()

    with get_db() as conn:
        production = conn.execute("""
            SELECT *
            FROM production_predictions
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()

        detections = conn.execute("""
            SELECT *
            FROM weed_detections
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()

    return render_template(
        "history.html",
        production=production,
        detections=detections
    )


# --------------------------------------------------
# 9. START FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    init_db()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
