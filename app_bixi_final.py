import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import datetime as dt
import pydeck as pdk

# -------------------------------------------------
# GLOBAL PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="BIXI Demand Analytics Dashboard", layout="wide")

# =================================================
# MODEL 1 HELPERS (prediction + history)
# =================================================
@st.cache_resource
def load_model1_assets():
    """Loads Model 1 models + meta once."""
    mlr_pipe = joblib.load("model1_mlr_pipeline.pkl")
    rf_model = joblib.load("model1_rf.pkl")
    with open("model1_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return mlr_pipe, rf_model, meta


@st.cache_resource
def load_model1_df():
    """Loads BIXI_MODEL data for Model 1 history plots."""
    df = pd.read_parquet("BIXI_MODEL.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df = df[df["month"].between(5, 10)].reset_index(drop=True)
    return df


def build_features_model1(
    station_name,
    date,
    hour,
    temperature,
    wind_speed,
    bad_weather,
    is_holiday,
    feels_like,
    meta,
):
    temp_mean = meta["temp_mean"]
    temp_std = meta["temp_std"]
    wind_mean = meta["wind_mean"]
    wind_std = meta["wind_std"]

    station_to_code = meta["station_to_code"]
    station_to_lat = meta["station_to_lat"]
    station_to_lon = meta["station_to_lon"]

    month = date.month
    day_of_week = date.weekday() + 1
    is_weekend = int(day_of_week in [5, 6])

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    if feels_like is None:
        feels_like = temperature - 0.7 * wind_speed

    temperature_scaled = (temperature - temp_mean) / temp_std if temp_std != 0 else 0.0
    wind_speed_scaled = (wind_speed - wind_mean) / wind_std if wind_std != 0 else 0.0

    temp_hour = temperature_scaled * hour
    temperature_sq = temperature_scaled ** 2
    temp_feels_interaction = temperature_scaled * feels_like

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

    weekend_hour_interaction = is_weekend * hour_sin

    lat = station_to_lat.get(station_name, 0.0)
    lon = station_to_lon.get(station_name, 0.0)
    station_encoded = station_to_code.get(station_name, 0)

    return {
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


def render_model1_page():
    mlr_pipe, rf_model, meta = load_model1_assets()
    model_df = load_model1_df()

    stations = meta["stations"]
    numerical_features_m1 = meta["numerical_features_m1"]
    feature_columns_m1 = meta["feature_columns_m1"]
    numerical_features_rf_m1 = meta["numerical_features_rf_m1"]
    feature_columns_rf_m1 = meta["feature_columns_rf_m1"]

    ALLOWED_MONTHS = [5, 6, 7, 8, 9, 10]

    st.header("🚲 BIXI Demand – Model 1: prediction & demand history")

    tab_pred, tab_hist = st.tabs(["Prediction", "Historical temperature vs demand"])

    # ------------------ Prediction tab ------------------
    with tab_pred:
        st.subheader("Demand prediction (Model 1)")

        model_choice = st.radio(
            "Select model to use:",
            ["Linear Regression (pipeline)", "Random Forest"],
            index=1,
            key="model1_model_choice",
        )

        col1, col2 = st.columns(2)
        with col1:
            station_name = st.selectbox("Station", stations, key="model1_station")
        with col2:
            date_input = st.date_input(
                "Date", dt.date(2024, 7, 15), key="model1_date"
            )

        hour = st.slider("Hour of day", 0, 23, 12, key="model1_hour")

        selected_month = date_input.month
        if selected_month not in ALLOWED_MONTHS:
            st.error(
                f"Selected month = {selected_month}. "
                "The model is only valid for months 5–10 (May–October)."
            )

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            temperature = st.number_input(
                "Temperature (°C)", -20.0, 40.0, 20.0, key="model1_temp"
            )
        with col_w2:
            wind_speed = st.number_input(
                "Wind speed (km/h)", 0.0, 80.0, 10.0, key="model1_wind"
            )
        with col_w3:
            feels_like = st.number_input(
                "Feels like (°C)",
                -20.0,
                40.0,
                value=float(20.0),
                key="model1_feels",
            )

        col_flags = st.columns(3)
        with col_flags[0]:
            bad_weather = st.selectbox(
                "Bad weather? (0=No, 1=Yes)", options=[0, 1], index=0, key="model1_bad"
            )
        with col_flags[1]:
            is_holiday = st.selectbox(
                "Holiday? (0=No, 1=Yes)", options=[0, 1], index=0, key="model1_hol"
            )
        with col_flags[2]:
            st.caption("`day_of_week` and `is_weekend` are computed from the date.")

        feature_values = build_features_model1(
            station_name=station_name,
            date=date_input,
            hour=hour,
            temperature=temperature,
            wind_speed=wind_speed,
            bad_weather=int(bad_weather),
            is_holiday=int(is_holiday),
            feels_like=feels_like,
            meta=meta,
        )

        if st.button("Predict hourly demand", key="model1_predict_button"):
            if selected_month not in ALLOWED_MONTHS:
                st.error(
                    "Cannot predict: selected date is outside May–October. "
                    "Choose a date with month 5–10."
                )
            else:
                if model_choice == "Linear Regression (pipeline)":
                    row = {col: None for col in feature_columns_m1}
                    for col in numerical_features_m1:
                        row[col] = feature_values[col]
                    row["station"] = station_name
                    X_input = pd.DataFrame([row], columns=feature_columns_m1)
                    pred = float(mlr_pipe.predict(X_input)[0])
                    label = "Linear Regression (pipeline)"
                else:
                    row = {col: None for col in feature_columns_rf_m1}
                    for col in numerical_features_rf_m1:
                        row[col] = feature_values[col]
                    row["station_encoded"] = feature_values["station_encoded"]
                    X_input = pd.DataFrame([row], columns=feature_columns_rf_m1)
                    pred = float(rf_model.predict(X_input)[0])
                    label = "Random Forest"

                pred = max(pred, 0.0)
                st.success(f"Predicted hourly demand ({label}): **{pred:.2f} trips**")

                with st.expander("Show engineered features sent to the model"):
                    st.json(feature_values)

    # ------------------ History tab ------------------
    with tab_hist:
        st.subheader("Historical temperature vs demand for a day")

        station_hist = st.selectbox("Station", meta["stations"], key="hist_station")

        station_dates = (
            model_df.loc[model_df["station"] == station_hist, "date"].dropna().unique()
        )
        station_dates = sorted(station_dates)

        if not station_dates:
            st.warning("No historical data found for this station.")
            return

        selected_date = st.selectbox(
            "Select a date (from historical data)",
            options=station_dates,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="hist_date",
        )

        df_day = model_df[
            (model_df["station"] == station_hist)
            & (model_df["date"] == selected_date)
        ].copy()

        if df_day.empty:
            st.warning("No data for this station on the selected date.")
        else:
            df_day = df_day.sort_values("hour")
            plot_df = df_day[["hour", "temperature", "total_demand"]].set_index("hour")
            st.markdown(
                f"**Station:** {station_hist}  |  **Date:** {selected_date:%Y-%m-%d}"
            )
            st.line_chart(plot_df)
            st.caption(
                "The chart shows actual hourly demand and temperature "
                "for the selected station and date."
            )


# =================================================
# MODEL 2 HELPERS (prediction with historical avgs)
# =================================================
@st.cache_resource
def load_model2_assets():
    mlr_pipe = joblib.load("model2_mlr_pipeline.pkl")
    rf_model = joblib.load("model2_rf.pkl")
    with open("bixi_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return mlr_pipe, rf_model, meta


def build_features_model2(
    station_name,
    date,
    hour,
    temperature,
    wind_speed,
    bad_weather,
    is_holiday,
    feels_like,
    meta,
):
    temp_mean = meta["temp_mean"]
    temp_std = meta["temp_std"]
    wind_mean = meta["wind_mean"]
    wind_std = meta["wind_std"]

    station_to_code = meta["station_to_code"]
    station_to_lat = meta["station_to_lat"]
    station_to_lon = meta["station_to_lon"]
    hourly_avg_lookup = meta["hourly_avg_lookup"]
    dow_avg_lookup = meta["dow_avg_lookup"]
    global_hourly_mean = meta["global_hourly_mean"]
    global_dow_mean = meta["global_dow_mean"]

    month = date.month
    day_of_week = date.weekday() + 1
    is_weekend = int(day_of_week in [5, 6])

    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    if feels_like is None:
        feels_like = temperature - 0.7 * wind_speed

    temperature_scaled = (temperature - temp_mean) / temp_std if temp_std != 0 else 0.0
    wind_speed_scaled = (wind_speed - wind_mean) / wind_std if wind_std != 0 else 0.0

    temperature_sq = temperature_scaled ** 2
    temp_hour = temperature_scaled * hour
    temp_feels_interaction = temperature_scaled * feels_like

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

    weekend_hour_interaction = is_weekend * hour_sin

    lat = station_to_lat.get(station_name, 0.0)
    lon = station_to_lon.get(station_name, 0.0)

    key_hour = (station_name, hour)
    key_dow = (station_name, day_of_week)
    avg_hourly_demand_station = hourly_avg_lookup.get(key_hour, global_hourly_mean)
    avg_dayofweek_station = dow_avg_lookup.get(key_dow, global_dow_mean)

    station_encoded = station_to_code.get(station_name, 0)

    return {
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


def render_model2_page():
    mlr_pipe, rf_model, meta = load_model2_assets()

    stations = meta["stations"]
    numerical_features_m2 = meta["numerical_features_m2"]
    feature_columns_m2 = meta["feature_columns_m2"]
    numerical_features_rf_m2 = meta["numerical_features_rf_m2"]
    feature_columns_rf_m2 = meta["feature_columns_rf_m2"]

    st.header("BIXI Demand – Model 2 prediction (with historical averages)")

    model_choice = st.radio(
        "Select model to use:",
        ["Model 2 – Linear Regression (pipeline)", "Model 2 – Random Forest"],
        index=1,
        key="model2_model_choice",
    )

    station_name = st.selectbox("Station", stations, key="model2_station")

    min_date = dt.date(2024, 5, 1)
    max_date = dt.date(2024, 10, 31)
    date_input = st.date_input(
        "Date (only May–October 2024 are allowed)",
        value=dt.date(2024, 7, 15),
        min_value=min_date,
        max_value=max_date,
        key="model2_date",
    )

    hour = st.slider("Hour of day", 0, 23, 12, key="model2_hour")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        temperature = st.number_input(
            "Temperature (°C)", -20.0, 40.0, 20.0, key="model2_temp"
        )
    with col_w2:
        wind_speed = st.number_input(
            "Wind speed (km/h)", 0.0, 80.0, 10.0, key="model2_wind"
        )
    with col_w3:
        feels_like = st.number_input(
            "Feels like (°C)",
            -20.0,
            40.0,
            value=float(20.0),
            key="model2_feels",
            help="If set different from Temperature, it is used in the temp–feels_like interaction.",
        )

    col_flags = st.columns(3)
    with col_flags[0]:
        bad_weather = st.selectbox(
            "Bad weather? (0=No, 1=Yes)", options=[0, 1], index=0, key="model2_bad"
        )
    with col_flags[1]:
        is_holiday = st.selectbox(
            "Holiday? (0=No, 1=Yes)", options=[0, 1], index=0, key="model2_hol"
        )
    with col_flags[2]:
        st.caption("`day_of_week` and `is_weekend` are computed from the date.")

    feature_values = build_features_model2(
        station_name=station_name,
        date=date_input,
        hour=hour,
        temperature=temperature,
        wind_speed=wind_speed,
        bad_weather=int(bad_weather),
        is_holiday=int(is_holiday),
        feels_like=feels_like,
        meta=meta,
    )

    if st.button("Predict hourly demand", key="model2_predict_button"):
        if model_choice == "Model 2 – Linear Regression (pipeline)":
            row = {col: None for col in feature_columns_m2}
            for col in numerical_features_m2:
                row[col] = feature_values[col]
            row["station"] = station_name
            X_input = pd.DataFrame([row], columns=feature_columns_m2)
            pred = float(mlr_pipe.predict(X_input)[0])
            label = "Model 2 – Linear Regression (pipeline)"
        else:
            row = {col: None for col in feature_columns_rf_m2}
            for col in numerical_features_rf_m2:
                row[col] = feature_values[col]
            row["station_encoded"] = feature_values["station_encoded"]
            X_input = pd.DataFrame([row], columns=feature_columns_rf_m2)
            pred = float(rf_model.predict(X_input)[0])
            label = "Model 2 – Random Forest"

        pred = max(pred, 0.0)
        st.success(f"Predicted hourly demand ({label}): **{pred:.2f} trips**")

        with st.expander("Show engineered features sent to the model"):
            st.json(feature_values)


# =================================================
# CLUSTERS PAGE (Model 1)
# =================================================
@st.cache_resource
def load_station_clusters():
    return pd.read_csv("station_clusters_model1.csv")


def render_clusters_page():
    clusters_df = load_station_clusters()

    required_cols = {"station", "lat", "lon", "mean_demand", "cluster_label"}
    missing = required_cols - set(clusters_df.columns)
    if missing:
        st.error(f"Missing columns in station_clusters_model1.csv: {missing}")
        return

    st.header("BIXI Station Demand Clusters (Model 1)")

    st.write(
        "Stations are grouped into **3 clusters** based on their average hourly demand: "
        "`low`, `medium`, and `high`. The heatmap below shows where higher-demand "
        "and lower-demand stations are located in the city."
    )

    cluster_choice = st.radio(
        "Which cluster do you want to visualize?",
        ["all", "low", "medium", "high"],
        index=0,
        key="cluster_choice",
    )

    if cluster_choice == "all":
        df_plot = clusters_df.copy()
        st.subheader("All clusters (low, medium, high)")
    else:
        df_plot = clusters_df[clusters_df["cluster_label"] == cluster_choice].copy()
        st.subheader(f"Cluster: {cluster_choice.capitalize()} demand")

    st.caption(f"Stations shown: {len(df_plot)}")

    color_map = {
        "low": [0, 120, 255],      # blue-ish
        "medium": [255, 215, 0],   # yellow-ish
        "high": [255, 0, 0],       # red
    }
    df_plot["color"] = df_plot["cluster_label"].map(color_map)

    if len(df_plot) > 0:
        view_state = pdk.ViewState(
            latitude=df_plot["lat"].mean(),
            longitude=df_plot["lon"].mean(),
            zoom=11,
            pitch=45,
        )

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_plot,
            get_position="[lon, lat]",
            get_weight="mean_demand",
            radiusPixels=60,
            aggregation='"SUM"',
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_plot,
            get_position="[lon, lat]",
            get_radius=40,
            get_fill_color="color",
            pickable=True,
        )

        tooltip = {
            "html": (
                "<b>Station:</b> {station}<br/>"
                "<b>Cluster:</b> {cluster_label}<br/>"
                "<b>Mean demand:</b> {mean_demand}"
            ),
            "style": {"color": "white"},
        }

        deck = pdk.Deck(
            layers=[heatmap_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
        )

        st.pydeck_chart(deck)
    else:
        st.warning("No stations to display for this cluster filter.")

    st.markdown("### Cluster summary")
    summary = (
        clusters_df.groupby("cluster_label")["mean_demand"]
        .agg(["count", "mean", "min", "max"])
        .rename(
            columns={
                "count": "num_stations",
                "mean": "avg_mean_demand",
                "min": "min_mean_demand",
                "max": "max_mean_demand",
            }
        )
        .reset_index()
        .sort_values("avg_mean_demand")
    )
    st.dataframe(summary, use_container_width=True)


# =================================================
# MAIN: SIDEBAR ROUTER
# =================================================
st.sidebar.title("Select view:")
view = st.sidebar.radio(
    "",
    [
        "Model 1 – prediction & demand history",
        "Model 2 – prediction with historical averages",
        "Clusters – station demand",
    ],
    index=0,
)

st.title("🚲 BIXI Demand Analytics Dashboard")

if view == "Model 1 – prediction & demand history":
    render_model1_page()
elif view == "Model 2 – prediction with historical averages":
    render_model2_page()
else:
    render_clusters_page()
