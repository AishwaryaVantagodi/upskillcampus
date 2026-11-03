"""
Train a custom crop/weed detector using a YOLO-format labelled dataset.

Expected dataset layout (edit data.yaml paths to match your actual files):
dataset/weed_detection/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml

YOLO labels are one .txt per image, with each row:
class_id x_center y_center width height
Coordinates are normalized to 0..1. Use class IDs consistently (e.g. 0=crop, 1=weed).
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "weed_detection" / "data.yaml"
if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Missing {DATA_YAML}. Prepare and label the crop/weed image dataset first."
    )

model = YOLO("yolo11n.pt")  # downloads pretrained base weights on first run
model.train(data=str(DATA_YAML), epochs=50, imgsz=512, project=str(ROOT / "runs"), name="crop_weed")
best = ROOT / "runs" / "crop_weed" / "weights" / "best.pt"
if best.exists():
    target_dir = ROOT / "models"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / "crop_weed_best.pt"
    target.write_bytes(best.read_bytes())
    print(f"Custom crop/weed detector saved to: {target}")
else:
    print("Training finished, but best.pt was not found at the expected path. Check the Ultralytics output.")
