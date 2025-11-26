import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import datetime as dt
import pydeck as pdk

# -------------------------------------------------
# Global page config
# -------------------------------------------------
st.set_page_config(page_title="BIXI Analytics Dashboard", layout="wide")

# -------------------------------------------------
# Sidebar navigation
# -------------------------------------------------
st.title("🚲 BIXI Demand Analytics Dashboard")
page = st.sidebar.radio(
    "Select view:",
    [
        "Model 1 – prediction & demand history",
        "Model 2 – prediction with historical averages",
        "Clusters – station demand",
    ],
    index=0,
)

# -------------------------------------------------
# Shared / cached loaders
# -------------------------------------------------

@st.cache_resource
def load_model1_artifacts():
    mlr_pipe = joblib.load("model1_mlr_pipeline.pkl")
    rf_model = joblib.load("model1_rf.pkl")
    with open("model1_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return mlr_pipe, rf_model, meta


@st.cache_resource
def load_model1_df():
    """
    Load BIXI_MODEL.csv used to train Model 1.
    Used only for the historical plots (not for prediction).
    """
    df = pd.read_parquet("BIXI_MODEL.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    # Ensure month / hour / date exist
    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    # Keep only May–October to match your modeling period
    df = df[df["month"].between(5, 10)].reset_index(drop=True)
    return df


@st.cache_resource
def load_model2_artifacts():
    mlr_pipe = joblib.load("model2_mlr_pipeline.pkl")
    rf_model = joblib.load("model2_rf.pkl")
    with open("bixi_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return mlr_pipe, rf_model, meta


@st.cache_resource
def load_station_clusters():
    df = pd.read_csv("station_clusters_model1.csv")
    return df


# =========================================================
# PAGE 1: MODEL 1 – PREDICTION & HISTORICAL PLOT
# =========================================================
if page == "Model 1 – prediction & history":
    mlr_pipe, rf_model, meta = load_model1_artifacts()
    model_df = load_model1_df()

    stations = meta["stations"]
    station_to_code = meta["station_to_code"]
    station_to_lat  = meta["station_to_lat"]
    station_to_lon  = meta["station_to_lon"]

    numerical_features_m1    = meta["numerical_features_m1"]
    feature_columns_m1       = meta["feature_columns_m1"]
    numerical_features_rf_m1 = meta["numerical_features_rf_m1"]
    feature_columns_rf_m1    = meta["feature_columns_rf_m1"]

    temp_mean = meta["temp_mean"]
    temp_std  = meta["temp_std"]
    wind_mean = meta["wind_mean"]
    wind_std  = meta["wind_std"]

    ALLOWED_MONTHS = [5, 6, 7, 8, 9, 10]  # May–October only

    # -----------------------------
    # Helper: feature engineering from raw inputs
    # (mirroring your EDA / model_df construction)
    # -----------------------------
    def build_feature_dict_from_raw(
        station_name,
        date,
        hour,
        temperature,
        wind_speed,
        bad_weather,
        is_holiday,
        feels_like=None,
    ):
        # 1) Base calendar features
        month = date.month
        # In your pipeline: day_of_week = dt.dayofweek + 1  (1–7, Mon=1)
        day_of_week = date.weekday() + 1
        # is_weekend = 1 if day_of_week in [5,6]
        is_weekend = int(day_of_week in [5, 6])

        # 2) Hour encodings
        hour_rad = 2 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # 3) Weather scaling – same idea as StandardScaler you used
        if feels_like is None:
            # your EDA: feels_like = temperature - 0.7 * wind_speed
            feels_like = temperature - 0.7 * wind_speed

        temperature_scaled = (temperature - temp_mean) / temp_std if temp_std != 0 else 0.0
        wind_speed_scaled  = (wind_speed - wind_mean) / wind_std if wind_std != 0 else 0.0

        # 4) Interaction terms (from your code)
        temp_hour = temperature_scaled * hour
        temperature_sq = temperature_scaled ** 2
        temp_feels_interaction = temperature_scaled * feels_like

        # 5) Hour bucket:
        # pd.cut(hour, bins=[-1,5,10,16,19,24],
        #        labels=['night','morning','day','evening','late'])
        if hour <= 5:
            hour_bucket_label = "night"
        elif hour <= 10:
            hour_bucket_label = "morning"
        elif hour <= 16:
            hour_bucket_label = "day"
        elif hour <= 19:
            hour_bucket_label = "evening"
        else:
            hour_bucket_label = "late"

        # Map labels → same codes as cat.codes (night=0, morning=1, day=2, evening=3, late=4)
        bucket_map = {"night": 0, "morning": 1, "day": 2, "evening": 3, "late": 4}
        hour_bucket = bucket_map[hour_bucket_label]

        # 6) Weekend-hour interaction
        weekend_hour_interaction = is_weekend * hour_sin

        # 7) Lat/lon from station (NOT asked from user)
        lat = station_to_lat.get(station_name, 0.0)
        lon = station_to_lon.get(station_name, 0.0)

        # Encoded station
        station_encoded = station_to_code.get(station_name, 0)

        feat = {
            "month": month,
            "day_of_week": day_of_week,
            "temperature_scaled": temperature_scaled,
            "wind_speed_scaled": wind_speed_scaled,
            "bad_weather": bad_weather,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "feels_like": feels_like,
            "is_holiday": is_holiday,
            "temp_hour": temp_hour,
            "temperature_sq": temperature_sq,
            "temp_feels_interaction": temp_feels_interaction,
            "hour_bucket": hour_bucket,
            "weekend_hour_interaction": weekend_hour_interaction,
            "lat": lat,
            "lon": lon,
            "station": station_name,
            "station_encoded": station_encoded,
        }

        return feat

    # -----------------------------
    # LAYOUT: two tabs
    # -----------------------------
    st.title("BIXI Hourly Demand – Model 1")

    tab_pred, tab_hist = st.tabs(["Prediction", "Historical temperature vs demand"])

    # =========================================================
    # TAB 1: PREDICTION (unchanged logic)
    # =========================================================
    with tab_pred:
        st.subheader("Demand prediction")

        st.write(
            "This tab serves your **Model 1** (Multiple Linear Regression and Random Forest). "
            "You select a station, date, hour, and raw weather conditions. "
            "The app internally reconstructs all engineered features "
            "(`temperature_scaled`, `hour_sin`, `hour_bucket`, etc.) "
            "using the same logic as when you built `BIXI_MODEL.csv`.\n\n"
            "**Note:** Predictions are only valid for months **May to October (5–10)**, "
            "because the model was trained on those months."
        )

        model_choice = st.radio(
            "Select model to use:",
            ["Linear Regression (pipeline)", "Random Forest"],
            index=1,
            key="model_choice_pred",
        )

        st.markdown("### 1. Station & time")

        station_name = st.selectbox("Station", stations, key="station_pred")

        col_date, col_hour = st.columns(2)
        with col_date:
            date_input = st.date_input("Date", dt.date(2024, 7, 15), key="date_pred")
        with col_hour:
            hour = st.slider("Hour of day", 0, 23, 12, key="hour_pred")

        # Check if selected month is allowed
        selected_month = date_input.month
        if selected_month not in ALLOWED_MONTHS:
            st.error(
                f"Selected month = {selected_month}. "
                "The model is only valid for months **5 to 10** (May–October). "
                "Please choose a date within that range."
            )

        st.markdown("### 2. Weather & context")

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            temperature = st.number_input("Temperature (°C)", -20.0, 40.0, 20.0, key="temp_pred")
        with col_w2:
            wind_speed = st.number_input("Wind speed (km/h)", 0.0, 80.0, 10.0, key="wind_pred")
        with col_w3:
            feels_like = st.number_input(
                "Feels like (°C)",
                -20.0,
                40.0,
                value=float(20.0),
                key="feels_pred",
                help="Used in the temp–feels_like interaction; "
                     "you can set it equal to temperature or adjust for wind/chill."
            )

        col_flags = st.columns(3)
        with col_flags[0]:
            bad_weather = st.selectbox("Bad weather? (0=No, 1=Yes)", options=[0, 1], index=0, key="bad_pred")
        with col_flags[1]:
            is_holiday = st.selectbox("Holiday? (0=No, 1=Yes)", options=[0, 1], index=0, key="holiday_pred")
        with col_flags[2]:
            st.caption("`day_of_week` and `is_weekend` are computed automatically from the date.")

        # Build features from raw inputs
        feature_values = build_feature_dict_from_raw(
            station_name=station_name,
            date=date_input,
            hour=hour,
            temperature=temperature,
            wind_speed=wind_speed,
            bad_weather=int(bad_weather),
            is_holiday=int(is_holiday),
            feels_like=feels_like,
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        if st.button("Predict hourly demand", key="predict_button"):
            if selected_month not in ALLOWED_MONTHS:
                st.error(
                    "Cannot predict: the selected date is outside the allowed months "
                    "(May–October). Please choose a date with month 5, 6, 7, 8, 9, or 10."
                )
            else:
                if model_choice == "Linear Regression (pipeline)":
                    # Build DataFrame with correct columns for the pipeline
                    row = {col: None for col in feature_columns_m1}
                    for col in numerical_features_m1:
                        row[col] = feature_values[col]
                    row["station"] = station_name

                    X_input = pd.DataFrame([row], columns=feature_columns_m1)

                    pred = float(mlr_pipe.predict(X_input)[0])
                    model_label = "Linear Regression (pipeline)"

                else:  # Random Forest
                    row = {col: None for col in feature_columns_rf_m1}
                    for col in numerical_features_rf_m1:
                        row[col] = feature_values[col]
                    row["station_encoded"] = feature_values["station_encoded"]

                    X_input = pd.DataFrame([row], columns=feature_columns_rf_m1)

                    pred = float(rf_model.predict(X_input)[0])
                    model_label = "Random Forest"

                pred = max(pred, 0.0)  # demand can't be negative
                st.success(f"Predicted hourly demand ({model_label}): **{pred:.2f} trips**")

                with st.expander("Show engineered features sent to the model"):
                    st.json(feature_values)

    # =========================================================
    # TAB 2: HISTORICAL TEMPERATURE vs DEMAND
    # =========================================================
    with tab_hist:
        st.subheader("Historical temperature vs demand for a day")

        st.write(
            "This tab uses your **BIXI_MODEL.csv** data to show **actual hourly demand and temperature** "
            "for a selected station and date between **May and October**."
        )

        # Station selection
        station_hist = st.selectbox("Station", stations, key="station_hist")

        # Available dates for this station (from the data)
        station_dates = (
            model_df.loc[model_df["station"] == station_hist, "date"]
            .dropna()
            .unique()
        )
        station_dates = sorted(station_dates)

        if len(station_dates) == 0:
            st.warning("No historical data found for this station in BIXI_MODEL.csv.")
        else:
            # Let the user pick one of the available dates
            default_date = station_dates[0]
            selected_date = st.selectbox(
                "Select a date (from historical data)",
                options=station_dates,
                format_func=lambda d: d.strftime("%Y-%m-%d"),
                index=0,
                key="date_hist",
            )

            # Filter data for that station + date
            df_day = model_df[
                (model_df["station"] == station_hist) &
                (model_df["date"] == selected_date)
            ].copy()

            if df_day.empty:
                st.warning("No data for this station on the selected date.")
            else:
                df_day = df_day.sort_values("hour")

                # Build a DataFrame with hour, temperature, total_demand
                plot_df = df_day[["hour", "temperature", "total_demand"]].copy()
                plot_df = plot_df.set_index("hour").sort_index()

                st.markdown(
                    f"**Station:** {station_hist}  |  **Date:** {selected_date.strftime('%Y-%m-%d')}"
                )

                st.line_chart(plot_df)

                st.caption(
                    "The chart shows **actual hourly demand** and **temperature** for the selected station and date. "
                    "Use it to visually explore how ridership responds to temperature across the day."
                )

# =========================================================
# PAGE 2: MODEL 2 – PREDICTION
# =========================================================
elif page == "Model 2 – prediction":
    mlr_pipe, rf_model, meta = load_model2_artifacts()

    stations = meta["stations"]
    station_to_code = meta["station_to_code"]
    station_to_lat  = meta["station_to_lat"]
    station_to_lon  = meta["station_to_lon"]

    numerical_features_m2    = meta["numerical_features_m2"]
    feature_columns_m2       = meta["feature_columns_m2"]
    numerical_features_rf_m2 = meta["numerical_features_rf_m2"]
    feature_columns_rf_m2    = meta["feature_columns_rf_m2"]

    temp_mean = meta["temp_mean"]
    temp_std  = meta["temp_std"]
    wind_mean = meta["wind_mean"]
    wind_std  = meta["wind_std"]

    hourly_avg_lookup   = meta["hourly_avg_lookup"]
    dow_avg_lookup      = meta["dow_avg_lookup"]
    global_hourly_mean  = meta["global_hourly_mean"]
    global_dow_mean     = meta["global_dow_mean"]

    # -----------------------------
    # Helper: feature engineering from raw inputs
    # (mirrors your EDA logic for BIXI_MODEL.csv)
    # -----------------------------
    def build_feature_dict_from_raw_m2(
        station_name: str,
        date: dt.date,
        hour: int,
        temperature: float,
        wind_speed: float,
        bad_weather: int,
        is_holiday: int,
        feels_like: float | None = None,
    ):
        # 1) Calendar features
        month = date.month
        # In your pipeline: day_of_week = dt.dayofweek + 1 (1–7, Mon=1)
        day_of_week = date.weekday() + 1
        is_weekend  = int(day_of_week in [5, 6])  # Fri/Sat as weekend in your code

        # 2) Hour encodings
        hour_rad = 2 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # 3) Weather scaling (same stats as training)
        if feels_like is None:
            feels_like = temperature - 0.7 * wind_speed  # your EDA definition

        temperature_scaled = (temperature - temp_mean) / temp_std if temp_std != 0 else 0.0
        wind_speed_scaled  = (wind_speed - wind_mean) / wind_std if wind_std != 0 else 0.0

        # 4) Interaction & nonlinear terms
        temperature_sq        = temperature_scaled ** 2
        temp_hour             = temperature_scaled * hour
        temp_feels_interaction = temperature_scaled * feels_like

        # 5) Hour bucket: same bins & labels as pd.cut([-1,5,10,16,19,24], labels)
        if hour <= 5:
            hour_bucket_label = "night"
        elif hour <= 10:
            hour_bucket_label = "morning"
        elif hour <= 16:
            hour_bucket_label = "day"
        elif hour <= 19:
            hour_bucket_label = "evening"
        else:
            hour_bucket_label = "late"

        bucket_map = {"night": 0, "morning": 1, "day": 2, "evening": 3, "late": 4}
        hour_bucket = bucket_map[hour_bucket_label]

        # 6) Weekend-hour interaction
        weekend_hour_interaction = is_weekend * hour_sin

        # 7) Lat/lon from station
        lat = station_to_lat.get(station_name, 0.0)
        lon = station_to_lon.get(station_name, 0.0)

        # 8) Historical averages (Model 2 features)
        # Use precomputed lookup tables with global fallback.
        key_hour = (station_name, hour)
        key_dow  = (station_name, day_of_week)

        avg_hourly_demand_station = hourly_avg_lookup.get(key_hour, global_hourly_mean)
        avg_dayofweek_station     = dow_avg_lookup.get(key_dow,  global_dow_mean)

        # 9) Encoded station for RF
        station_encoded = station_to_code.get(station_name, 0)

        feat = {
            "month": month,
            "day_of_week": day_of_week,
            "temperature_scaled": temperature_scaled,
            "wind_speed_scaled": wind_speed_scaled,
            "bad_weather": bad_weather,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "feels_like": feels_like,
            "is_holiday": is_holiday,
            "temp_hour": temp_hour,
            "temperature_sq": temperature_sq,
            "temp_feels_interaction": temp_feels_interaction,
            "hour_bucket": hour_bucket,
            "weekend_hour_interaction": weekend_hour_interaction,
            "lat": lat,
            "lon": lon,
            "avg_hourly_demand_station": avg_hourly_demand_station,
            "avg_dayofweek_station": avg_dayofweek_station,
            "station": station_name,
            "station_encoded": station_encoded,
        }

        return feat

    # -----------------------------
    # UI
    # -----------------------------
    st.title("BIXI Hourly Demand Prediction – Model 2 (with historical averages)")

    st.write(
        "This page serves **Model 2** (Multiple Linear Regression and Random Forest) "
        "trained on `BIXI_MODEL.csv`. You select a station, a date (from May to October 2024), "
        "an hour, and raw weather conditions. The app reconstructs all engineered features "
        "(`temperature_scaled`, `hour_sin`, `hour_bucket`, historical averages, etc.) "
        "exactly as in your notebook."
    )

    model_choice = st.radio(
        "Select model to use:",
        ["Model 2 – Linear Regression (pipeline)", "Model 2 – Random Forest"],
        index=1
    )

    st.markdown("### 1. Station & time")

    station_name = st.selectbox("Station", stations)

    # Only allow months 5–10 (May–October 2024)
    min_date = dt.date(2024, 5, 1)
    max_date = dt.date(2024, 10, 31)

    date_input = st.date_input(
        "Date (only May–October 2024 are allowed)",
        value=dt.date(2024, 7, 15),
        min_value=min_date,
        max_value=max_date,
    )

    hour = st.slider("Hour of day", 0, 23, 12)

    st.markdown("### 2. Weather & context")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        temperature = st.number_input("Temperature (°C)", -20.0, 40.0, 20.0)
    with col_w2:
        wind_speed = st.number_input("Wind speed (km/h)", 0.0, 80.0, 10.0)
    with col_w3:
        feels_like = st.number_input(
            "Feels like (°C)",
            -20.0,
            40.0,
            value=float(20.0),
            help="If you set this different from Temperature, it's used in the temp–feels_like interaction."
        )

    col_flags = st.columns(3)
    with col_flags[0]:
        bad_weather = st.selectbox("Bad weather? (0=No, 1=Yes)", options=[0, 1], index=0)
    with col_flags[1]:
        is_holiday = st.selectbox("Holiday? (0=No, 1=Yes)", options=[0, 1], index=0)
    with col_flags[2]:
        st.caption("`day_of_week` and `is_weekend` are computed automatically from the date.")

    # -----------------------------
    # Build features from raw
    # -----------------------------
    feature_values = build_feature_dict_from_raw_m2(
        station_name=station_name,
        date=date_input,
        hour=hour,
        temperature=temperature,
        wind_speed=wind_speed,
        bad_weather=int(bad_weather),
        is_holiday=int(is_holiday),
        feels_like=feels_like,
    )

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict hourly demand"):
        if model_choice == "Model 2 – Linear Regression (pipeline)":
            # Build DataFrame with correct columns for Model 2 MLR
            row = {col: None for col in feature_columns_m2}
            for col in numerical_features_m2:
                row[col] = feature_values[col]
            row["station"] = station_name

            X_input = pd.DataFrame([row], columns=feature_columns_m2)

            pred = float(mlr_pipe.predict(X_input)[0])
            model_label = "Model 2 – Linear Regression (pipeline)"

        else:  # Model 2 RF
            row = {col: None for col in feature_columns_rf_m2}
            for col in numerical_features_rf_m2:
                row[col] = feature_values[col]
            row["station_encoded"] = feature_values["station_encoded"]

            X_input = pd.DataFrame([row], columns=feature_columns_rf_m2)

            pred = float(rf_model.predict(X_input)[0])
            model_label = "Model 2 – Random Forest"

        pred = max(pred, 0.0)  # demand cannot be negative
        st.success(f"Predicted hourly demand ({model_label}): **{pred:.2f} trips**")

        with st.expander("Show engineered features sent to the model"):
            st.json(feature_values)

# =========================================================
# PAGE 3: CLUSTERS – STATION DEMAND
# =========================================================
else:  # "Clusters – station demand"
    clusters_df = load_station_clusters()

    # Basic sanity
    required_cols = {"station", "lat", "lon", "mean_demand", "cluster_label"}
    missing = required_cols - set(clusters_df.columns)
    if missing:
        st.error(f"Missing columns in station_clusters_model1.csv: {missing}")
    else:
        # -----------------------------
        # UI
        # -----------------------------
        st.title("BIXI Station Demand Clusters (Model 1)")

        st.write(
            "Stations are grouped into **3 clusters** based on their average hourly demand: "
            "`low`, `medium`, and `high`. The heatmap below shows where higher-demand "
            "and lower-demand stations are located in the city."
        )

        cluster_choice = st.radio(
            "Which cluster do you want to visualize?",
            ["all", "low", "medium", "high"],
            index=0
        )

        if cluster_choice == "all":
            df_plot = clusters_df.copy()
            st.subheader("All clusters (low, medium, high)")
        else:
            df_plot = clusters_df[clusters_df["cluster_label"] == cluster_choice].copy()
            st.subheader(f"Cluster: {cluster_choice.capitalize()} demand")

        st.caption(f"Stations shown: {len(df_plot)}")

        # -----------------------------
        # Color mapping per cluster
        # low  = blue-ish
        # med  = yellow-ish
        # high = red
        # -----------------------------
        color_map = {
            "low":    [0, 120, 255],   # blue-ish
            "medium": [255, 215, 0],   # yellow-ish
            "high":   [255, 0, 0],     # red
        }

        # Assign a color to each station row based on its cluster_label
        df_plot["color"] = df_plot["cluster_label"].map(color_map)

        # -----------------------------
        # Heatmap (pydeck)
        # -----------------------------
        if len(df_plot) > 0:
            # Center the map around the mean lat/lon
            view_state = pdk.ViewState(
                latitude=df_plot["lat"].mean(),
                longitude=df_plot["lon"].mean(),
                zoom=11,
                pitch=45,
            )

            # Heatmap layer: weight = mean_demand
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=df_plot,
                get_position="[lon, lat]",
                get_weight="mean_demand",
                radiusPixels=60,
                aggregation='"SUM"',
            )

            # Scatter layer to see station points (color by cluster)
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_plot,
                get_position="[lon, lat]",
                get_radius=40,
                get_fill_color="color",   # uses the 'color' column we just created
                pickable=True,
            )

            tooltip = {
                "html": (
                    "<b>Station:</b> {station}<br/>"
                    "<b>Cluster:</b> {cluster_label}<br/>"
                    "<b>Mean demand:</b> {mean_demand}"
                ),
                "style": {"color": "white"}
            }

            deck = pdk.Deck(
                layers=[heatmap_layer, scatter_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )

            st.pydeck_chart(deck)
        else:
            st.warning("No stations to display for this cluster filter.")

        # -----------------------------
        # Summary table / explainability
        # -----------------------------
        st.markdown("### Cluster summary")

        summary = (
            clusters_df
            .groupby("cluster_label")["mean_demand"]
            .agg(["count", "mean", "min", "max"])
            .rename(columns={
                "count": "num_stations",
                "mean": "avg_mean_demand",
                "min": "min_mean_demand",
                "max": "max_mean_demand",
            })
            .reset_index()
            .sort_values("avg_mean_demand")
        )

        st.dataframe(summary, use_container_width=True)

        st.markdown(
            """
            **Interpretation:**
            - `low` cluster → stations with the lowest average hourly demand (blue)  
            - `medium` cluster → mid-range stations (yellow)  
            - `high` cluster → busiest stations (red)  

            In the **All clusters** view, you can visually compare where low, medium, and high
            demand stations are throughout the city.
            The heatmap color intensity reflects station density *and* their average demand.
            """
        )
