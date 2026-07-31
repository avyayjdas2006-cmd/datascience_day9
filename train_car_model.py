"""
Train the car-price prediction model.

This reuses the modeling approach from the previous assignment notebook
(notebooks/car_price_prediction_4_.ipynb): one-hot encode the categorical
columns, scale all features, and train a RandomForestRegressor, which was
identified there as the best of three candidates (Linear Regression,
Decision Tree, Random Forest) with R^2 ~ 0.887.

The only change from the notebook is packaging: instead of manual
pd.get_dummies() + a separately-saved StandardScaler (which breaks on a
brand/model value that wasn't in the training data), everything is wrapped
in a single sklearn Pipeline with a OneHotEncoder(handle_unknown="ignore").
That makes the artifact self-contained and safe to call directly from the
Streamlit app with raw form inputs.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/cardekho_dataset.csv"
MODEL_PATH = "models/car_price_model.pkl"
META_PATH = "models/car_price_meta.json"

CATEGORICAL = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]
NUMERIC = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
TARGET = "selling_price"


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "car_name"], errors="ignore")
    df = df.dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scale", StandardScaler(with_mean=False)),
            ("model", RandomForestRegressor(
                random_state=42, n_estimators=100, max_depth=20,
                min_samples_leaf=1, n_jobs=-1,
            )),
        ]
    )


def main():
    df = load_and_clean(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"Held-out test set (20%, n={len(y_test)}):")
    print(f"  MAE : {mae:,.0f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  R^2 : {r2:.4f}")

    # Refit on the full dataset for the deployed model (standard practice:
    # once architecture + hyperparameters are validated on a held-out split,
    # ship a final model trained on all available data).
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH, compress=3)

    meta = {
        "brands": sorted(df["brand"].unique().tolist()),
        "models_by_brand": {
            b: sorted(g["model"].unique().tolist())
            for b, g in df.groupby("brand")
        },
        "seller_types": sorted(df["seller_type"].unique().tolist()),
        "fuel_types": sorted(df["fuel_type"].unique().tolist()),
        "transmission_types": sorted(df["transmission_type"].unique().tolist()),
        "seats_options": sorted(int(s) for s in df["seats"].unique() if s > 0),
        "ranges": {
            "vehicle_age": [int(df["vehicle_age"].min()), int(df["vehicle_age"].max())],
            "km_driven": [int(df["km_driven"].min()), int(df["km_driven"].max())],
            "mileage": [float(df["mileage"].min()), float(df["mileage"].max())],
            "engine": [int(df["engine"].min()), int(df["engine"].max())],
            "max_power": [float(df["max_power"].min()), float(df["max_power"].max())],
        },
        "test_metrics": {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(y_test)},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved UI metadata to {META_PATH}")


if __name__ == "__main__":
    main()
