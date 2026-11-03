from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "processed_data" / "cleaned_production.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Run `python preprocess.py` first.")

    df = pd.read_csv(DATA_FILE)
    required = {"Particulars", "Year", "Production"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required}; got {df.columns.tolist()}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df["Particulars"] = df["Particulars"].astype(str).str.strip()
    df = df.dropna(subset=["Particulars", "Year", "Production"])
    df["Year"] = df["Year"].astype(int)

    # Each category needs multiple observations; split by time, not randomly.
    latest_year = int(df["Year"].max())
    test_start = max(int(df["Year"].min()) + 1, latest_year - 2)
    train = df[df["Year"] < test_start].copy()
    test = df[df["Year"] >= test_start].copy()

    if train.empty or test.empty or train["Year"].nunique() < 2:
        raise ValueError("Not enough distinct years for a time-based evaluation.")

    X_train = train[["Particulars", "Year"]]
    y_train = train["Production"]
    X_test = test[["Particulars", "Year"]]
    y_test = test["Production"]

    prep = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), ["Particulars"]),
        ("year", "passthrough", ["Year"])
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds) if len(y_test) > 1 else float("nan")
    print(f"Time-based test years: {sorted(test['Year'].unique().tolist())}")
    print(f"MAE: {mae:.4f} (same units as source data)")
    print(f"RMSE: {rmse:.4f} (same units as source data)")
    print(f"R²: {r2:.4f}")

    joblib.dump(model, MODEL_DIR / "production_model.pkl")
    joblib.dump(sorted(df["Particulars"].unique().tolist()), MODEL_DIR / "categories.pkl")
    # Keep the source unit associated with each category for honest display.
    if "Unit" in df.columns:
        units = df[["Particulars", "Unit"]].drop_duplicates("Particulars")
        unit_map = dict(zip(units["Particulars"], units["Unit"].fillna("unit not specified")))
    else:
        unit_map = {}
    joblib.dump(unit_map, MODEL_DIR / "units.pkl")
    joblib.dump({"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}, MODEL_DIR / "metrics.pkl")
    print("Saved production model and category metadata to models/")

if __name__ == "__main__":
    main()
