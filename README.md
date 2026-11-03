# Integrated Smart Agriculture AI (Project 4 + Project 5)

This starter project combines:
1. **Project 4 — Prediction of Agricultural Crop Production in India** using the supplied historical production CSV, preprocessing, a baseline Linear Regression model, and a Flask interface.
2. **Project 5 — Crop and Weed Detection** using a custom YOLO object detection model and a Flask upload page.

## Important limitations
- The included CSV has `Particulars`, `Frequency`, `Unit`, and annual values. It does not provide all the fields in the original project description (e.g., state, district, variety, cost, recommended zone).
- The production model estimates historical category values from year and category. Future estimates are extrapolations and not official forecasts.
- **The custom crop/weed model is not included**, because the labelled image dataset was not supplied. The detection page will explain this until you add a labelled YOLO dataset and train the model.

## Setup (Windows / VS Code)
Open a terminal in this folder:

```bash
python -m venv venv
venv\\Scripts\\activate
python -m pip install -r requirements.txt
python preprocess.py
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Enable Project 5
1. Put your actual labelled images and YOLO labels into `dataset/weed_detection/`.
2. Create/edit `dataset/weed_detection/data.yaml` (see its README).
3. Train with `python train_yolo.py`.
4. The script should save `models/crop_weed_best.pt`.
5. Restart Flask and use the Crop & Weed Detection page.

YOLO training can require a GPU for practical speed. The initial model weights may download from the internet. Do not describe the detector as working until it has been trained and tested on labelled images.

## Notes
- `agriculture.db` is created automatically to store prediction and detection history.
- For a public deployment, set a secure secret key, limit upload size, and do not use Flask debug mode.
