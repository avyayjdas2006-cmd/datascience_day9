# ML Predictor Suite — Car Price & Customer Churn

**Participant Name:** Avyay J Das
**MUID:** avyayjdas@mulearn

## 🌐 Deployment Link

> _Add your live Streamlit Community Cloud URL here after deploying, e.g._
> `https://<your-app-name>.streamlit.app`

---

## 📖 Project Overview

A single Streamlit app that bundles two independent ML predictors, built as two
"pages" of one deployable application:

| App | Task | Model | Held-out performance |
|---|---|---|---|
| 🚗 Car Price Predictor | Regression — estimate a used car's resale price (₹) | Random Forest Regressor | R² = 0.888, MAE ≈ ₹99,000 |
| 📉 Customer Churn Predictor | Classification — predict whether a subscriber will churn | Decision Tree Classifier (max_depth=10) | Accuracy = 99.8%, ROC AUC = 0.999 |

Both models were **reused from a previous assignment** (per the brief's
"reuse or retrain" option) and then **retrained** inside clean, self-contained
scikit-learn `Pipeline`s so they can be deployed reliably:

- **Car price** — reuses the exact approach from
  `notebooks/car_price_prediction_4_.ipynb` (one-hot encode categoricals →
  scale → Random Forest, which that notebook identified as the best of three
  candidate models). See *Deployment Approach* below for the one packaging
  change made.
- **Customer churn** — reuses the architecture and hyperparameters of the
  pretrained model supplied from the earlier assignment
  (`notebooks/best_churn_model_original.pkl`, a
  `DecisionTreeClassifier(max_depth=10, random_state=42)`), retrained
  end-to-end because the original scaler/training script weren't available
  (see *Challenges Faced*).

## 🗂️ Project Structure

```
.
├── app.py                              # Home page / entry point
├── common.py                           # Shared CSS + UI helpers
├── pages/
│   ├── 1_Car_Price_Predictor.py        # Car price mini-app
│   └── 2_Customer_Churn_Predictor.py   # Churn mini-app
├── train_car_model.py                  # Trains + saves the car price pipeline
├── train_churn_model.py                # Trains + saves the churn pipeline
├── models/
│   ├── car_price_model.pkl             # Trained sklearn Pipeline
│   ├── car_price_meta.json             # Dropdown options + ranges for the form
│   ├── churn_model.pkl
│   └── churn_meta.json
├── data/
│   ├── cardekho_dataset.csv
│   └── customer_churn_dataset.csv
├── notebooks/                          # Original assignment artifacts (reference)
│   ├── car_price_prediction_4_.ipynb
│   └── best_churn_model_original.pkl
├── requirements.txt
└── README.md
```

## 🚀 Deployment Approach

1. **Model packaging.** Both models are wrapped in a single
   `sklearn.pipeline.Pipeline` (`ColumnTransformer` with `OneHotEncoder` →
   model), saved with `joblib.dump(..., compress=3)`. This is the one change
   from the original notebook's approach, which used manual
   `pd.get_dummies()` plus a separately fitted `StandardScaler` — convenient
   for a notebook, but fragile in an app where a user's form selection needs
   to be encoded exactly the same way at request time. A single pipeline
   object takes a raw one-row DataFrame straight from the form and returns a
   prediction, with `handle_unknown="ignore"` so an unseen category can't
   crash the app.
2. **Interface.** One Streamlit app with a home page and two
   auto-discovered sub-pages (`pages/`), so it's one repo and one deployment
   that serves both predictors, navigable via `st.page_link`.
3. **Hosting.** Built for **Streamlit Community Cloud** (free, and the most
   direct path from a public GitHub repo to a public URL). To deploy:
   1. Push this folder to a public GitHub repository.
   2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
      GitHub, and click **New app**.
   3. Pick the repo/branch and set the main file path to `app.py`.
   4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
      automatically and gives you a public `*.streamlit.app` URL.
   5. Paste that URL into the *Deployment Link* section at the top of this
      README.
4. **Reproducing the models.** The `.pkl` files in `models/` are already
   trained and committed, so the app works out of the box. To retrain:
   ```bash
   pip install -r requirements.txt
   python train_car_model.py
   python train_churn_model.py
   streamlit run app.py
   ```

## 🔍 Key Observations

- **Encoding, not scaling, mattered most for the tree models.** Both the
  Decision Tree (churn) and Random Forest (price) are invariant to
  monotonic scaling of individual features, so the real engineering work was
  getting the *categorical encoding* exactly right and reproducible, not
  the numeric scaling.
- **Brand/model is the dominant price signal.** Re-running the car pipeline
  reproduced the notebook's own finding almost exactly (R² 0.888 here vs.
  0.887 in the notebook) — confirming the packaging change didn't alter
  model behavior, just how it's called.
- **Retraining beat reusing the raw artifact for churn.** The supplied
  `best_churn_model_original.pkl` is a real, valid `DecisionTreeClassifier`,
  but calling it directly against the only data file available produced
  near-random accuracy (~47–54%, ROC AUC ≈ 0.5–0.56 under several
  reasonable encoding/scaling guesses) — see *Challenges Faced*.

## 🧗 Challenges Faced

- **Missing preprocessing artifacts for the churn model.** Only the trained
  `.pkl` was provided from the previous assignment — not the training
  notebook, the fitted encoder/scaler, or the original training data.
  Inspecting `model.feature_names_in_` recovered the exact feature schema
  (label-encoded `Gender`, one-hot dummies for `Subscription Type` and
  `Contract Length` with `Basic`/`Annual` as the dropped baselines), but
  without the original scaler, predictions on the one churn CSV available
  were essentially uncorrelated with the true labels. Rather than ship a
  model that looks reused but performs at chance, the same architecture and
  hyperparameters were retrained end-to-end on the available data — accuracy
  went from ~50% to 99.8%, which is itself a good illustration of why
  "reuse" and "retrain" are both legitimate, and sometimes retraining is the
  more honest choice.
- **High-cardinality categoricals for the car model.** `brand` (32 values)
  and `model` (120 values) one-hot encode to ~160 columns. This kept the
  Random Forest's default settings from being deployable (an uncompressed,
  untuned forest came out to ~240 MB, over GitHub's per-file limit).
  `n_estimators`/`max_depth` were tuned down and `joblib` compression was
  applied, cutting the file to ~20 MB with no meaningful loss in R².
- **Streamlit's multipage navigation.** `st.page_link` only resolves page
  metadata correctly when the app is entered through its main script
  (`app.py`) — verified with Streamlit's `AppTest` harness by driving
  navigation through the entry point rather than instantiating a sub-page
  in isolation.

## 🔮 Future Improvements

- Add SHAP-based explanations so a prediction comes with "why" (e.g. which
  features pushed a customer's churn risk up).
- Track prediction history / feedback in a small database to monitor drift
  once real users start using the app.
- For the car price model, engineer a smarter `model` feature (e.g. group
  rare models into an "other" bucket per brand) to reduce dimensionality
  without losing signal.
- Add authentication + a lightweight admin view for retraining/re-uploading
  models without a redeploy.
- Containerize with Docker for hosting options beyond Streamlit Community
  Cloud (Render, Railway, Hugging Face Spaces) with more control over
  resources.
