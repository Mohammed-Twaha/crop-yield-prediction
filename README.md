# Crop Yield Prediction — India (State/District Level)

🔗 **Live App:** https://crop-yield-prediction-2dc8zwkgwdiotjxqbqu698.streamlit.app/

A data science portfolio project predicting agricultural crop yield (tonnes/hectare) across Indian states and districts, using historical crop production data from 1997–2020.

## Problem Statement

Given a **State, District, Crop, Season, and Year**, predict the expected crop yield in tonnes per hectare. This kind of model helps illustrate how historical agricultural patterns (region, crop choice, season) drive yield outcomes — useful for planning, benchmarking, and resource allocation discussions.

## Dataset

- **Source:** State/district-level crop production statistics for India (1997–2020), compiled from India's Ministry of Agriculture & Farmers Welfare open data (mirrored from the Kaggle "Crop Production in India" dataset).
- **Size:** ~326,000 rows after cleaning
- **Coverage:** 37 states/UTs, 707 districts, 54 crops, 6 seasons, 24 years
- **Columns:** `state`, `district`, `crop`, `year`, `season`, `area` (hectares), `production` (tonnes), `yield` (tonnes/hectare)

## Approach

1. **Data cleaning** — removed invalid rows (zero area/yield), capped extreme per-crop outliers at the 0.5th/99.5th percentile.
2. **EDA** — yield trends over time, top states/crops by yield, seasonal patterns, area-production relationship, correlation analysis.
3. **Feature engineering** — deliberately excluded `area`/`production` as model inputs (since `yield = production / area` by definition, using them would leak the target). Instead, the model predicts yield purely from **State, District (frequency-encoded), Crop, Season, and Year** — matching exactly what a user would input in the deployed app.
4. **Modeling** — compared Linear Regression (baseline), Random Forest, and XGBoost.
5. **Deployment** — best model (XGBoost) exported and served via a Streamlit app.

## Results

| Model | RMSE (t/ha) | MAE (t/ha) | R² |
|---|---|---|---|
| Linear Regression | 5.95 | 2.45 | 0.766 |
| Random Forest | 3.12 | 1.06 | 0.935 |
| **XGBoost (best)** | **2.90** | **0.99** | **0.944** |

XGBoost explains ~94% of the variance in yield using only categorical/temporal context — no weather data required. Crop type and state are the dominant predictors (see `outputs/feature_importance.png`).

## Project Structure

```
crop_yield/
├── data/
│   └── main_crops.csv              # Raw dataset
├── notebook/
│   └── crop_yield_prediction.ipynb # Full EDA + modeling notebook (executed, with outputs)
├── models/
│   ├── crop_yield_model.pkl        # Trained XGBoost pipeline
│   ├── district_freq_map.pkl       # District frequency encoding map
│   └── app_meta.pkl                # Dropdown options + metrics for the app
├── app/
│   └── app.py                      # Streamlit deployment app
├── outputs/                        # EDA & model comparison charts (PNG)
├── requirements.txt
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt

# Explore the notebook
jupyter notebook notebook/crop_yield_prediction.ipynb

# Run the app
cd app
streamlit run app.py
```

## Deploying the App

Push this repo to GitHub, then deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Connect your GitHub repo
2. Set the main file path to `app/app.py`
3. Deploy — done.

## Limitations & Future Work

- **No weather data.** Yield is heavily driven by rainfall and temperature, which aren't in this dataset. Merging in IMD district-level rainfall data is the single biggest lever to improve accuracy further.
- **No lag features.** Adding previous-year yield for the same state+crop+season would likely improve short-term predictions.
- **No soil/irrigation data.** Would help disambiguate districts with similar names/climates but different farming infrastructure.

---

**Submitted by:** Mohammed Twaha

**Role:** Data Science Intern, Zephyr

**GitHub:** https://github.com/Mohammed-Twaha

**LinkedIn:** https://linkedin.com/in/mohammed-twaha-010320203
