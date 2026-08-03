# 🚗 Used Car Price Predictor

**Participant Name:** Avyay J Das
**MUID:** avyayjdas@mulearn


A Streamlit web app that estimates the resale value of a used car, built on a
CarDekho-style dataset. A `scikit-learn` pipeline (preprocessing + a
`RandomForestRegressor`) is trained offline and served through an interactive
UI where a user fills in a car's details and gets an instant price estimate.

## Features

- Interactive sidebar form for brand, model, age, kilometers driven, seller
  type, fuel type, transmission, mileage, engine size, max power, and seats
- Real-time prediction with an estimated price range (±10%)
- Model metadata (dropdown options, valid ranges, held-out MAE / R²) is
  generated once at training time and reused by the app — no need to touch
  the raw data at inference time
- Single deployable model artifact (`model/car_price_model.pkl`) that bundles
  all preprocessing (`StandardScaler` + `OneHotEncoder`) with the estimator

## Project Structure

```
.
├── app.py                      # Streamlit app (UI + inference)
├── train_model.py              # Trains and saves the model pipeline
├── generate_synthetic_data.py  # Generates a schema-matched synthetic dataset
├── requirements.txt
├── .gitignore
├── data/
│   └── cardekho_dataset.csv    # (generated) training data
└── model/
    ├── car_price_model.pkl     # (generated) trained pipeline
    └── metadata.json           # (generated) UI dropdown options, ranges, metrics
```

## About the Data

This sandbox can't reach Kaggle directly, so `generate_synthetic_data.py`
produces a synthetic dataset that mirrors the **schema** and general price
relationships of the real
[CarDekho Used Car Price dataset](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data)
(same columns: `brand, model, vehicle_age, km_driven, seller_type, fuel_type,
transmission_type, mileage, engine, max_power, seats, selling_price`).

For production-grade accuracy, download the real Kaggle CSV, save it as
`data/cardekho_dataset.csv` with matching column names, and re-run
`train_model.py` — no changes needed in `app.py`.

## Setup

```bash
# 1. Clone and enter the project
git clone <this-repo-url>
cd <this-repo>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**1. Generate the training data** (skip this step if you're using the real
Kaggle CSV instead — just place it at `data/cardekho_dataset.csv`):

```bash
python generate_synthetic_data.py
```

**2. Train the model** — cleans the data, fits the pipeline, and writes
`model/car_price_model.pkl` and `model/metadata.json`:

```bash
python train_model.py
```

**3. Run the app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## How It Works

- **Preprocessing:** numeric features (`vehicle_age`, `km_driven`, `mileage`,
  `engine`, `max_power`, `seats`) are standard-scaled; categorical features
  (`brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`) are
  one-hot encoded via a `ColumnTransformer`.
- **Model:** `RandomForestRegressor` (120 trees, max depth 12, min leaf size
  3), wrapped together with the preprocessor in a single `sklearn.Pipeline`
  so `app.py` only ever calls `.predict()` on raw input.
- **Cleaning:** rows with fewer than 2 seats, more than 500,000 km driven, or
  non-positive selling price are dropped before training.
- **Evaluation:** an 80/20 train/test split reports MAE and R² on held-out
  data; both are saved to `metadata.json` and shown in the app's "About this
  model" section.

## Notes & Limitations

- The bundled model is trained on **synthetic** data and is meant to
  demonstrate the end-to-end pipeline, not to price real cars accurately.
- Swap in the real CarDekho CSV and retrain for meaningful predictions.
- Predicted prices are clipped at 0 and shown with a ±10% range as a rough
  confidence band, not a statistical prediction interval.
