"""
train_model.py
---------------------------------
Trains a regression model to predict used-car selling price (CarDekho-style
dataset) and saves a single deployable artifact: model/car_price_model.pkl

Usage:
    python train_model.py

The artifact is a scikit-learn Pipeline (preprocessing + model), so app.py
never has to re-implement encoding logic — it just calls `.predict()`.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/cardekho_dataset.csv"
MODEL_PATH = "model/car_price_model.pkl"
META_PATH = "model/metadata.json"

TARGET = "selling_price"
NUMERIC_FEATURES = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CATEGORICAL_FEATURES = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])

    # --- light data cleaning (real-world CarDekho data has a few bad rows) ---
    before = len(df)
    df = df[df["seats"] >= 2]                 # drop 0-seat entry errors
    df = df[df["km_driven"] <= 500_000]        # drop extreme km outliers (e.g. 3.8M km)
    df = df[df["selling_price"] > 0]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows during cleaning (out of {before}).")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE:  Rs. {mae:,.0f}")
    print(f"Test R^2:  {r2:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

    # Save dropdown choices + ranges + metrics so the Streamlit app can build
    # its inputs without touching the raw CSV again.
    brand_models = (
        df.groupby("brand")["model"].unique().apply(lambda x: sorted(x.tolist())).to_dict()
    )
    metadata = {
        "brand_models": brand_models,
        "seller_type": sorted(df["seller_type"].unique().tolist()),
        "fuel_type": sorted(df["fuel_type"].unique().tolist()),
        "transmission_type": sorted(df["transmission_type"].unique().tolist()),
        "seats_options": sorted(int(s) for s in df["seats"].unique().tolist()),
        "ranges": {
            "vehicle_age": [0, int(df["vehicle_age"].max())],
            "km_driven": [0, int(df["km_driven"].max())],
            "mileage": [float(df["mileage"].min()), float(df["mileage"].max())],
            "engine": [int(df["engine"].min()), int(df["engine"].max())],
            "max_power": [float(df["max_power"].min()), float(df["max_power"].max())],
        },
        "metrics": {"mae": float(mae), "r2": float(r2), "n_rows": int(len(df))},
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
