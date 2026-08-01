"""
app.py
---------------------------------
Streamlit app: Used Car Price Predictor (CarDekho-style dataset).
Loads the trained pipeline from model/car_price_model.pkl and metadata.json,
collects user inputs, and returns a real-time price prediction.
"""

import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

MODEL_PATH = "model/car_price_model.pkl"
META_PATH = "model/metadata.json"


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, meta = load_artifacts()

st.title("🚗 Used Car Price Predictor")
st.write(
    "Estimate the resale value of a used car. Fill in the car's details on the "
    "left and get an instant, model-based price estimate."
)

with st.sidebar:
    st.header("Car Details")

    brand = st.selectbox("Brand", sorted(meta["brand_models"].keys()))
    model_options = meta["brand_models"][brand]
    model_name = st.selectbox("Model", model_options)

    age_min, age_max = meta["ranges"]["vehicle_age"]
    vehicle_age = st.slider("Vehicle Age (years)", int(age_min), int(age_max), 5)

    km_min, km_max = meta["ranges"]["km_driven"]
    km_driven = st.number_input(
        "Kilometers Driven", min_value=0, max_value=int(km_max), value=40000, step=1000
    )

    seller_type = st.selectbox("Seller Type", meta["seller_type"])
    fuel_type = st.selectbox("Fuel Type", meta["fuel_type"])
    transmission_type = st.selectbox("Transmission", meta["transmission_type"])

    mil_min, mil_max = meta["ranges"]["mileage"]
    mileage = st.slider(
        "Mileage (kmpl)", float(mil_min), float(mil_max), float((mil_min + mil_max) / 2)
    )

    eng_min, eng_max = meta["ranges"]["engine"]
    engine = st.number_input(
        "Engine (CC)", min_value=int(eng_min), max_value=int(eng_max), value=1200, step=50
    )

    pow_min, pow_max = meta["ranges"]["max_power"]
    max_power = st.slider(
        "Max Power (bhp)", float(pow_min), float(pow_max), float((pow_min + pow_max) / 2)
    )

    seats = st.selectbox("Seats", meta["seats_options"])

    predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame(
        [
            {
                "vehicle_age": vehicle_age,
                "km_driven": km_driven,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
                "brand": brand,
                "model": model_name,
                "seller_type": seller_type,
                "fuel_type": fuel_type,
                "transmission_type": transmission_type,
            }
        ]
    )

    prediction = pipeline.predict(input_df)[0]
    prediction = max(0, prediction)

    st.subheader("Estimated Selling Price")
    st.metric(label=f"{brand} {model_name}", value=f"₹ {prediction:,.0f}")
    st.caption(f"≈ ₹ {prediction / 100000:,.2f} Lakh")

    low, high = prediction * 0.9, prediction * 1.1
    st.write(f"Likely range: ₹ {low:,.0f} – ₹ {high:,.0f}")

    with st.expander("See the inputs used for this prediction"):
        st.dataframe(input_df, use_container_width=True)
else:
    st.info("Set the car's details in the sidebar, then click **Predict Price**.")

st.divider()
with st.expander("ℹ️ About this model"):
    st.write(
        f"Trained on {meta['metrics']['n_rows']:,} listings using a "
        "RandomForestRegressor inside a scikit-learn pipeline "
        "(StandardScaler + OneHotEncoder)."
    )
    st.write(
        f"Held-out test performance — MAE: ₹ {meta['metrics']['mae']:,.0f}, "
        f"R²: {meta['metrics']['r2']:.3f}."
    )
    st.caption(
        "Note: this demo instance is trained on a schema-matched synthetic "
        "dataset. Swap in the real CarDekho CSV (see README) and re-run "
        "train_model.py for production-grade accuracy."
    )
