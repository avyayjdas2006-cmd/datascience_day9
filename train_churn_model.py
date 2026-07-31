"""
Train the customer-churn prediction model.

A pretrained model (notebooks/best_churn_model_original.pkl) was supplied
from the previous assignment. Inspecting it shows it is a
DecisionTreeClassifier(max_depth=10, random_state=42) expecting 12 named
features: Age, Gender, Tenure, Usage Frequency, Support Calls,
Payment Delay, Total Spend, Last Interaction, and one-hot dummies for
Subscription Type and Contract Length.

However, only the standalone .pkl was provided -- not the original
training notebook/script or the fitted scaler it may have relied on, and
not the data file it was originally trained on. Reconstructing the
encoding from `feature_names_in_` and validating the pretrained model
against the one churn CSV we do have showed it did not generalize to that
file (near-chance ROC AUC), most likely because it was trained on a larger
file from the same public dataset with a different preprocessing pipeline
we can't fully recover.

Per the assignment brief ("reuse OR retrain a model from a previous
assignment"), we retrain: same algorithm and hyperparameters
(DecisionTreeClassifier, max_depth=10, random_state=42) that were used
before, refit end-to-end on the available data inside a single sklearn
Pipeline, so behavior is fully reproducible and safe to call from the app.
"""

import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/customer_churn_dataset.csv"
MODEL_PATH = "models/churn_model.pkl"
META_PATH = "models/churn_meta.json"

CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]
NUMERIC = [
    "Age", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Total Spend", "Last Interaction",
]
TARGET = "Churn"


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["CustomerID"], errors="ignore")
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
            ("model", DecisionTreeClassifier(max_depth=10, random_state=42)),
        ]
    )


def main():
    df = load_and_clean(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    print(f"Held-out test set (20%, n={len(y_test)}):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC AUC : {auc:.4f}")
    print(classification_report(y_test, preds))

    # Refit on all available data for the deployed model.
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH, compress=3)

    meta = {
        "genders": sorted(df["Gender"].unique().tolist()),
        "subscription_types": sorted(df["Subscription Type"].unique().tolist()),
        "contract_lengths": sorted(df["Contract Length"].unique().tolist()),
        "ranges": {
            "Age": [int(df["Age"].min()), int(df["Age"].max())],
            "Tenure": [int(df["Tenure"].min()), int(df["Tenure"].max())],
            "Usage Frequency": [int(df["Usage Frequency"].min()), int(df["Usage Frequency"].max())],
            "Support Calls": [int(df["Support Calls"].min()), int(df["Support Calls"].max())],
            "Payment Delay": [int(df["Payment Delay"].min()), int(df["Payment Delay"].max())],
            "Total Spend": [float(df["Total Spend"].min()), float(df["Total Spend"].max())],
            "Last Interaction": [int(df["Last Interaction"].min()), int(df["Last Interaction"].max())],
        },
        "test_metrics": {"accuracy": acc, "roc_auc": auc, "n_test": len(y_test), "report": report},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved UI metadata to {META_PATH}")


if __name__ == "__main__":
    main()
