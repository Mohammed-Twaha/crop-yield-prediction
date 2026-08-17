import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="India Crop Yield Predictor", page_icon="🌾", layout="centered")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "crop_yield_model.pkl"))
    district_freq = joblib.load(os.path.join(MODEL_DIR, "district_freq_map.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "app_meta.pkl"))
    return model, district_freq, meta

model, district_freq, meta = load_artifacts()

st.title("🌾 India Crop Yield Predictor")
st.markdown(
    "Predict crop yield (tonnes/hectare) for a given **state, district, crop, season, and year**, "
    "trained on India's crop production data (1997–2020)."
)

with st.sidebar:
    st.header("About")
    st.write(
        f"**Model:** {meta['best_model_name']}\n\n"
        f"**Test R²:** {meta['metrics']['R2']:.3f}\n\n"
        f"**Test RMSE:** {meta['metrics']['RMSE']:.2f} t/ha\n\n"
        "Trained on state/district-level crop production records across India, 1997–2020."
    )
    st.caption(
        "Note: this model uses only location, crop, season, and year as inputs — "
        "it does not have access to weather/rainfall data, so predictions represent "
        "a historical-pattern estimate, not a weather-adjusted forecast."
    )

st.subheader("Enter details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", meta["states"], index=meta["states"].index("Karnataka") if "Karnataka" in meta["states"] else 0)
    districts = meta["districts_by_state"].get(state, [])
    district = st.selectbox("District", districts)
    crop = st.selectbox("Crop", meta["crops"], index=meta["crops"].index("Rice") if "Rice" in meta["crops"] else 0)

with col2:
    season = st.selectbox("Season", meta["seasons"])
    year = st.slider("Year", min_value=meta["year_range"][0], max_value=meta["year_range"][1], value=2024)

if st.button("Predict Yield", type="primary", use_container_width=True):
    freq = district_freq.get(district, district_freq.mean())

    input_df = pd.DataFrame([{
        "state": state,
        "crop": crop,
        "season": season,
        "year": year,
        "district_freq": freq
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)  # yield can't be negative

    st.success(f"### Predicted Yield: **{prediction:.2f} tonnes/hectare**")
    st.caption(
        f"{crop} · {season} season · {district}, {state} · {year}"
    )

    st.markdown("---")
    st.markdown(
        "**Interpretation:** This estimate reflects historical yield patterns for this "
        "crop/region/season combination. Actual yield in any given year will vary with "
        "rainfall, temperature, pest pressure, and farming practices not captured here."
    )

st.markdown("---")
st.caption(
    "Built as a data science portfolio project · Data: India crop production statistics (1997–2020) · "
    "Models compared: Linear Regression, Random Forest, XGBoost"
)
