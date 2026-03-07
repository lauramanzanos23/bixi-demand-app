# BIXI Demand App

Streamlit application and modeling assets for predicting Montreal BIXI station demand and exploring station demand patterns.

## 1. Project overview
This repository delivers an end-to-end demand analytics project for BIXI bike-share usage:
- Data preparation and feature engineering for station-level hourly demand
- Two predictive model families for demand forecasting
- Station clustering to segment low/medium/high-demand locations
- A Streamlit dashboard to interact with predictions and historical demand patterns

Primary use case: estimate expected hourly trips for a station given date/time and weather conditions, and visualize demand behavior spatially across stations.

## 2. What the app does
The app entrypoint is `src/app.py` and provides three views:

1. **Model 1 – prediction & demand history**
- Predicts hourly demand for a selected station/date/hour using either:
  - Linear Regression pipeline, or
  - Random Forest
- Shows historical chart of temperature vs. demand by station/day
- Enforces seasonality guardrails for May-October usage window

2. **Model 2 – prediction with historical averages**
- Predicts hourly demand using weather/date features plus station historical-average features
- Supports the same two model families (Linear Regression pipeline and Random Forest)

3. **Clusters – station demand**
- Loads precomputed station cluster labels (`low`, `medium`, `high`)
- Shows map-based visualization and summary statistics by cluster

## 3. Visuals
### Average hourly demand profile
![Average BIXI demand by hour](docs/images/hourly-demand-profile.png)

### Station demand clusters
![BIXI station demand clusters](docs/images/station-demand-clusters.png)

## 4. Data and artifacts
### Core data files
- `data/processed/BIXI_MODEL.parquet`: model-ready hourly demand dataset
- `data/external/weatherstats_montreal_daily.csv`: weather history reference
- `data/external/mtl_weather_2024.csv`: weather data for project period

### Model artifacts
- `models/model1_mlr_pipeline.pkl`
- `models/model1_rf.pkl`
- `models/model1_meta.pkl`
- `models/model2_mlr_pipeline.pkl`
- `models/model2_rf.pkl`
- `models/bixi_meta.pkl`

### Clustering artifacts
- `artifacts/station_clusters_model1.csv`
- `artifacts/station_clusters_model1.pkl`

## 5. Modeling approach (high level)
### Model 1
- Uses temporal, station, weather, and engineered interaction features
- Includes cyclical hour encoding (`hour_sin`, `hour_cos`), weekend/holiday flags, and scaled weather variables
- Designed for demand prediction plus historical comparison workflows in the dashboard

### Model 2
- Extends feature set with historical station demand priors (hourly and day-of-week averages)
- Intended to improve station-specific calibration

### Cluster model
- Groups stations by average demand behavior into low/medium/high categories
- Used for exploratory geospatial segmentation in the app

## 6. Notebooks and workflow
The notebooks in `notebooks/` document training and analysis steps:
- `BIXI_Data_Cleaning.ipynb`: preparation and transformations
- `BIXI_Model_1.ipynb`: Model 1 training workflow
- `BIXI_Model_2.ipynb`: Model 2 training workflow
- `BIXI_Model_Clustering.ipynb`: station clustering workflow

In the current structure, notebooks are the source of model training logic; the app is inference-only.

## 7. Repository structure
```text
bixi-demand-app/
├── src/
│   └── app.py
├── models/
│   ├── model1_mlr_pipeline.pkl
│   ├── model1_rf.pkl
│   ├── model1_meta.pkl
│   ├── model2_mlr_pipeline.pkl
│   ├── model2_rf.pkl
│   └── bixi_meta.pkl
├── data/
│   ├── processed/
│   │   └── BIXI_MODEL.parquet
│   └── external/
│       ├── weatherstats_montreal_daily.csv
│       └── mtl_weather_2024.csv
├── artifacts/
│   ├── station_clusters_model1.csv
│   └── station_clusters_model1.pkl
├── notebooks/
│   ├── BIXI_Data_Cleaning.ipynb
│   ├── BIXI_Model_1.ipynb
│   ├── BIXI_Model_2.ipynb
│   └── BIXI_Model_Clustering.ipynb
├── docs/
│   └── images/
│       ├── hourly-demand-profile.png
│       └── station-demand-clusters.png
├── scripts/
│   ├── prueba.py
│   └── myclass.py
├── tests/
├── requirements.txt
├── runtime.txt
└── README.md
```

## 8. Setup and run
### Requirements
- Python `3.12` (see `runtime.txt`)
- Dependencies in `requirements.txt`

### Local run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

The app resolves data/model paths from the project root via `pathlib`, so it can be launched from different working directories.

## 9. Validation
Recommended checks after edits:

```bash
python -m compileall src scripts
streamlit run src/app.py --server.headless true
```

## 10. Dev container
`.devcontainer/devcontainer.json` is configured to:
- Open `README.md` and `src/app.py`
- Launch Streamlit with:

```bash
streamlit run src/app.py --server.enableCORS false --server.enableXsrfProtection false
```

## 11. Known limitations
- Training is notebook-driven; there is no single scripted training pipeline yet.
- Models are constrained by training period and assumptions (notably seasonal constraints in the app).
- Re-training and reproducibility controls (data versioning/ML experiment tracking) are not yet formalized.

## 12. Suggested next improvements
1. Add a `src/train/` package with reproducible CLI training scripts.
2. Add automated tests in `tests/` for feature builders and model inference shape checks.
3. Add model performance report (metrics + error slices) as a markdown artifact.
4. Add CI to run linting, tests, and basic app smoke checks on pull requests.
