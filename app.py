import json

import streamlit as st

from common import PALETTE, hero, inject_base_css

st.set_page_config(
    page_title="ML Predictor Suite",
    page_icon="🧭",
    layout="wide",
)
inject_base_css()

with open("models/car_price_meta.json") as f:
    car_meta = json.load(f)
with open("models/churn_meta.json") as f:
    churn_meta = json.load(f)

hero(
    "Two models, one repo",
    "ML Predictor Suite",
    "A small collection of deployed machine learning tools. Pick a predictor "
    "below, fill in a few details, and get an instant, model-backed estimate.",
)

st.write("")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        f"""
        <div class="app-card">
            <div class="eyebrow">Regression</div>
            <h3 style="margin-top:0;">🚗 Car Price Predictor</h3>
            <p style="color:{PALETTE['muted']};">
                Estimates the fair resale price of a used car from its brand,
                age, mileage, and specs — trained on {car_meta['test_metrics']['n_test']*5:,}
                real CarDekho listings.
            </p>
            <p><span class="eyebrow">Model</span><br/>Random Forest Regressor</p>
            <p><span class="eyebrow">Held-out R²</span><br/>{car_meta['test_metrics']['r2']:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Car_Price_Predictor.py", label="Open Car Price Predictor", icon="🚗")

with col2:
    st.markdown(
        f"""
        <div class="app-card">
            <div class="eyebrow">Classification</div>
            <h3 style="margin-top:0;">📉 Customer Churn Predictor</h3>
            <p style="color:{PALETTE['muted']};">
                Flags which subscribers are likely to churn based on usage,
                support calls, spend, and contract details — trained on
                {churn_meta['test_metrics']['n_test']*5:,} customer records.
            </p>
            <p><span class="eyebrow">Model</span><br/>Decision Tree Classifier</p>
            <p><span class="eyebrow">Held-out accuracy</span><br/>{churn_meta['test_metrics']['accuracy']:.1%}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Customer_Churn_Predictor.py", label="Open Churn Predictor", icon="📉")

st.write("")
st.markdown("---")
st.markdown(
    f"""
    <p style="color:{PALETTE['muted']}; font-size:0.9rem;">
    Built for the ML deployment assignment · both models are retrained versions of
    models developed in earlier assignments, wrapped in scikit-learn pipelines and
    served through this Streamlit app. See the README for the full write-up.
    </p>
    """,
    unsafe_allow_html=True,
)
