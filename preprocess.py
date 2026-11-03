from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "Production-Department_of_Agriculture_and_Cooperation_1.csv"
OUT_DIR = ROOT / "processed_data"
OUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place Production-Department_of_Agriculture_and_Cooperation_1.csv in the dataset folder."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.astype(str).str.strip()

    year_cols = [c for c in df.columns if c.isdigit() and len(c) == 4]
    if not year_cols:
        # This file uses headers such as '3-1993' after whitespace trimming.
        year_cols = [c for c in df.columns if c.startswith("3-") and c[2:].isdigit()]

    if not year_cols:
        raise ValueError(
            "Could not identify year columns. Expected columns like '3-2001' or '2001'. "
            f"Found: {df.columns.tolist()}"
        )

    id_cols = [c for c in ["Particulars", "Frequency", "Unit"] if c in df.columns]
    data = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="Year",
        value_name="Production"
    )
    data["Year"] = data["Year"].astype(str).str.extract(r"(\d{4})")[0]
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Production"] = (
        data["Production"].astype(str).str.replace(",", "", regex=False).str.strip()
    )
    data["Production"] = pd.to_numeric(data["Production"], errors="coerce")
    data = data.dropna(subset=["Year", "Production"])
    data["Year"] = data["Year"].astype(int)
    if "Particulars" in data.columns:
        data["Particulars"] = data["Particulars"].astype(str).str.strip()
    data = data.sort_values(["Particulars", "Year"]).reset_index(drop=True)

    output = OUT_DIR / "cleaned_production.csv"
    data.to_csv(output, index=False)
    print(f"Preprocessing complete: {len(data)} records")
    print(f"Saved: {output}")
    print("Columns:", data.columns.tolist())
    print(data.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
