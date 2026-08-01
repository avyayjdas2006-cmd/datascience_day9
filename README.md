 🚗 Used Car Price Predictor — ML Deployment Challenge

**Participant Name:** Avyay J Das
**MUID:** avyayjdas@mulearn

---

## 📌 Project Overview

This project takes a regression model trained on the **CarDekho Used Car
Price Prediction** dataset (Day 5,
[Kaggle link](https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data),
15,411 real listings) and deploys it as an interactive **Streamlit** web
application. A user enters a car's details — brand, model, age, kilometers
driven, fuel type, transmission, engine specs, etc. — and receives an
instant estimated resale price.

**Pipeline:**
1. `data/cardekho_dataset.csv` — the real Kaggle dataset (15,411 rows, 32
   brands, 120 models).
2. `train_model.py` — light cleaning (drops a couple of bad rows — see
   [Challenges Faced](#-challenges-faced)), then builds a `scikit-learn`
   `Pipeline` (`StandardScaler` + `OneHotEncoder` → `RandomForestRegressor`),
   trains it, evaluates it, and saves `model/car_price_model.pkl` +
   `model/metadata.json`.
3. `app.py` — the Streamlit UI. Loads the saved pipeline and lets users get
   live predictions with no retraining needed.

`generate_synthetic_data.py` is also included — it was used to bootstrap a
schema-matched dummy dataset while the real Kaggle CSV was being sourced,
so the project could be developed and tested end-to-end beforehand. It's
no longer needed now that the real data is in `data/`, but is kept for
reference / in case you want to regenerate a quick synthetic sample.

### Model performance (on held-out test split, real data)
| Metric | Value |
|---|---|
| MAE | ≈ ₹ 0.95 Lakh |
| R² | ≈ 0.938 |
| Training rows | 15,397 (after cleaning) |

## 🗂️ Project Structure
```
car-price-predictor/
├── app.py                      # Streamlit web app
├── train_model.py              # Trains and saves the model pipeline
├── generate_synthetic_data.py  # Builds the demo dataset
├── requirements.txt
├── README.md
├── data/
│   └── cardekho_dataset.csv
└── model/
    ├── car_price_model.pkl     # Trained pipeline (preprocessing + model)
    └── metadata.json           # Dropdown options, ranges, metrics for the UI
```

## 📊 About the Dataset

`data/cardekho_dataset.csv` is the real **CarDekho Used Car Price
Prediction** dataset from Kaggle — 15,411 listings across 32 brands and
120 models, with columns: `brand`, `model`, `vehicle_age`, `km_driven`,
`seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`,
`max_power`, `seats`, `selling_price`.

If you ever want to retrain on an updated export, just overwrite
`data/cardekho_dataset.csv` with the same column names and re-run
`python train_model.py` — `app.py` needs no changes since it reads whatever
`model/metadata.json` reports.

## 🚀 Running Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd car-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate data and retrain
python generate_synthetic_data.py
python train_model.py

# 4. Launch the app
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## 🌐 Deployment Approach

Deployed on **Streamlit Community Cloud** (free, and the simplest path from a
public GitHub repo to a live URL):

1. Push this project to a public GitHub repository (including the `model/`
   folder, so the app doesn't need to retrain on startup).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app** → select the repo, branch (`main`), and main file
   path (`app.py`).
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` and starts
   the app automatically. Every future push to `main` auto-redeploys.
5. Copy the generated public URL (`https://<app-name>.streamlit.app`) into
   the [Deployment Link](#-deployment-link) section below.

*Alternative platforms this repo also works on unchanged: Hugging Face
Spaces (Streamlit SDK), Render (Web Service, start command
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`), or
Railway.*

## 🌐 Deployment Link
> _[Paste your public app URL here after deploying, e.g.
> `https://your-app-name.streamlit.app`]_

## 🔍 Key Observations
- Wrapping preprocessing and the model together in a single `sklearn.Pipeline`
  made deployment much simpler — the app only ever calls `.predict()` on raw,
  human-readable inputs, with no manual encoding logic duplicated in `app.py`.
- Saving UI metadata (dropdown options, slider ranges, metrics) to a small
  `metadata.json` file at training time keeps the app in sync with the model
  automatically — no hardcoded lists to maintain by hand.
- `st.cache_resource` avoids reloading the model file on every user
  interaction, keeping predictions fast.
- On the real data, R² landed around 0.94 with MAE ≈ ₹95K, which is solid
  given selling prices range from tens of thousands to tens of lakhs.
  Vehicle age, kilometers driven, and engine/power specs were the strongest
  drivers of predicted price, matching domain intuition about depreciation.

## 🧩 Challenges Faced
- **Real-world data quality:** the raw CSV had a couple of bad rows — two
  listings with `seats = 0` (clearly a data-entry error for a Honda City
  and a Nissan Kicks) and one listing with `km_driven = 3,800,000` (a
  Mahindra XUV500 that's almost certainly missing a decimal point or has an
  extra digit). `train_model.py` filters these out (`seats >= 2`,
  `km_driven <= 500,000`) before training — without this, the outlier alone
  would have distorted the `km_driven` slider range in the app.
- Balancing input flexibility (letting users pick any brand/model
  combination) against realistic predictions required constraining the model
  dropdown to only the models seen for each selected brand.
- Keeping the model artifact and its preprocessing logic bundled as one
  `.pkl` avoided a common deployment bug: preprocessing code drifting out of
  sync with the trained model.
- The full 300-tree RandomForest first trained was ~44MB — too heavy for a
  clean GitHub push. Trimming to 120 trees / `max_depth=12` brought it down
  to ~14MB with a negligible accuracy trade-off.

## 🔮 Future Improvements
- Add outlier detection/handling that's more systematic than fixed
  thresholds (e.g. IQR-based filtering per brand).
- Add model comparison (e.g. Gradient Boosting vs Random Forest vs Linear
  Regression) with a leaderboard in the app.
- Add a feature-importance chart in the app so users can see *why* a price
  was predicted.
- Add input validation/warnings for unusual combinations (e.g. very high
  power with very low engine CC).
- Track prediction history in-session and let users compare multiple cars
  side by side.
- Add authentication/rate-limiting if deployed for wider public use, and
  monitoring for prediction drift over time.
