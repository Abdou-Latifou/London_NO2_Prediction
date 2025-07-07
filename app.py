from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import pyreadr
import geopandas as gpd
from pyproj import CRS, Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
import os
from datetime import datetime

app = Flask(__name__)

# Define CRS
crs_bng = CRS("+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.999601272 +x_0=400000 +y_0=-100000 +ellps=airy +units=km +no_defs")
crs_wgs84 = CRS("EPSG:4326")
transformer = Transformer.from_crs(crs_bng, crs_wgs84, always_xy=True)

# Global variables to store data and results
data = {"sitedata": None, "grid_data": None}
results = {"grid_data": None, "cv_summary": None}
years_available = []

# Serve static files (optional for background images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    global data, results, years_available
    debug_message = "Waiting for file uploads and model execution..."

    if request.method == "POST":
        # Handle file uploads
        sitedata_file = request.files.get("sitedata_file")
        griddata_file = request.files.get("griddata_file")
        model_choice = request.form.get("model")
        years = request.form.getlist("years")
        run_model = "run" in request.form

        if sitedata_file and griddata_file:
            try:
                # Save and load files
                upload_dir = os.path.join(app.root_path, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                sitedata_path = os.path.join(upload_dir, sitedata_file.filename)
                griddata_path = os.path.join(upload_dir, griddata_file.filename)
                sitedata_file.save(sitedata_path)
                griddata_file.save(griddata_path)

                sitedata = pyreadr.read_r(sitedata_path)["sitedata"]
                grid_data = pyreadr.read_r(griddata_path)["all"]

                # Validate and preprocess
                if "Year" not in sitedata.columns:
                    debug_message = "Error: 'Year' column not found in sitedata."
                    return render_template("index.html", debug_message=debug_message, years_available=years_available)
                if not pd.to_numeric(sitedata["Year"], errors="coerce").notnull().all():
                    debug_message = "Error: 'Year' column contains non-numeric values."
                    return render_template("index.html", debug_message=debug_message, years_available=years_available)

                sitedata["date"] = pd.to_datetime(sitedata["Year"].astype(float).astype(int).astype(str))
                sitedata["time"] = pd.factorize(sitedata["Year"])[0] + 1
                sitedata["log_NO2"] = np.log(sitedata["NO2"] + 1)
                sitedata = sitedata.dropna()

                # Ensure consistent column names and types
                grid_data = grid_data[["Easting", "Northing", "Time", "NDVI", "gridid", "Year"]]
                grid_data = grid_data.rename(columns={"Time": "time"})
                grid_data["Year"] = grid_data["Year"].astype(float)  # Ensure Year is float

                data = {"sitedata": sitedata, "grid_data": grid_data}
                years_available = sorted(sitedata["Year"].astype(float).unique())
                debug_message = f"Data loaded successfully. Years available: {', '.join(map(str, years_available))}"
            except Exception as e:
                debug_message = f"Error loading data: {str(e)}"
                return render_template("index.html", debug_message=debug_message, years_available=years_available)

        if run_model and data["sitedata"] is not None and years:
            try:
                debug_message += f"\nRunning {model_choice} model for years: {', '.join(years)}"
                sitedata = data["sitedata"]
                grid_data = data["grid_data"]

                if model_choice == "Random Forest":
                    train_data = sitedata[["log_NO2", "Easting", "Northing", "time", "NDVI"]]
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(train_data[["Easting", "Northing", "time", "NDVI"]], train_data["log_NO2"])

                    # Predictions
                    pred_data = grid_data[["Easting", "Northing", "time", "NDVI"]]
                    pred_all = rf_model.predict(pred_data)
                    grid_data["log_NO2_pred"] = pred_all
                    grid_data["log_NO2_sd"] = np.std([tree.predict(pred_data) for tree in rf_model.estimators_], axis=0)
                    grid_data["NO2_pred"] = np.exp(grid_data["log_NO2_pred"])
                    grid_data["NO2_sd"] = grid_data["NO2_pred"] * grid_data["log_NO2_sd"]

                    # Cross-validation
                    cv_results = []
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    for fold, (train_idx, test_idx) in enumerate(kf.split(sitedata)):
                        train_fold = sitedata.iloc[train_idx]
                        test_fold = sitedata.iloc[test_idx]
                        rf_fold = RandomForestRegressor(n_estimators=500, random_state=42)
                        rf_fold.fit(train_fold[["Easting", "Northing", "time", "NDVI"]], train_fold["log_NO2"])
                        pred_log = rf_fold.predict(test_fold[["Easting", "Northing", "time", "NDVI"]])
                        pred = np.exp(pred_log)
                        obs = test_fold["NO2"]
                        rmse = np.sqrt(mean_squared_error(obs, pred))
                        mae = mean_absolute_error(obs, pred)
                        bias = np.mean(pred - obs)
                        r2 = r2_score(obs, pred)
                        cv_results.append({"Fold": fold + 1, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2})
                    cv_summary = pd.DataFrame(cv_results).mean().to_dict()

                    results = {"grid_data": grid_data, "cv_summary": cv_summary}
                    debug_message += f"\n{model_choice} model completed."

                else:  # Gaussian Process (INLA-SPDE Proxy)
                    kernel = RBF(length_scale=10) + WhiteKernel(noise_level=1)
                    gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
                    train_data = sitedata[["log_NO2", "Easting", "Northing", "time", "NDVI"]]
                    gp_model.fit(train_data[["Easting", "Northing", "time", "NDVI"]], train_data["log_NO2"])

                    # Predictions
                    pred_data = grid_data[["Easting", "Northing", "time", "NDVI"]]
                    pred_all, pred_std = gp_model.predict(pred_data, return_std=True)
                    grid_data["log_NO2_pred"] = pred_all
                    grid_data["log_NO2_sd"] = pred_std
                    grid_data["NO2_pred"] = np.exp(grid_data["log_NO2_pred"])
                    grid_data["NO2_sd"] = grid_data["NO2_pred"] * grid_data["log_NO2_sd"]

                    # Cross-validation
                    cv_results = []
                    kf = KFold(n_splits=5, shuffle=True, random_state=23)
                    for fold, (train_idx, test_idx) in enumerate(kf.split(sitedata)):
                        train_fold = sitedata.iloc[train_idx]
                        test_fold = sitedata.iloc[test_idx]
                        gp_fold = GaussianProcessRegressor(kernel=kernel, random_state=42)
                        gp_fold.fit(train_fold[["Easting", "Northing", "time", "NDVI"]], train_fold["log_NO2"])
                        pred_log, _ = gp_fold.predict(test_fold[["Easting", "Northing", "time", "NDVI"]], return_std=True)
                        pred = np.exp(pred_log)
                        obs = test_fold["NO2"]
                        rmse = np.sqrt(mean_squared_error(obs, pred))
                        mae = mean_absolute_error(obs, pred)
                        bias = np.mean(pred - obs)
                        r2 = r2_score(obs, pred)
                        cv_results.append({"Fold": fold + 1, "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2})
                    cv_summary = pd.DataFrame(cv_results).mean().to_dict()

                    results = {"grid_data": grid_data, "cv_summary": cv_summary}
                    debug_message += f"\n{model_choice} model completed."
            except Exception as e:
                debug_message += f"\nError running model: {str(e)}"
                return render_template("index.html", debug_message=debug_message, years_available=years_available)

    # Generate plots
    pred_plot = None
    uncertainty_plot = None
    scatter_plot = None
    time_series_plot = None
    map_html = None
    error_message = ""

    if results["grid_data"] is not None and years:
        try:
            # Convert years to float for comparison
            selected_years = [float(y) for y in years]
            filtered_data = results["grid_data"][results["grid_data"]["Year"].isin(selected_years)]
            debug_message += f"\nFiltered data rows: {filtered_data.shape[0]}"
            # Clean data
            filtered_data = filtered_data.dropna(subset=["NO2_pred", "NO2_sd"])
            filtered_data = filtered_data[~np.isinf(filtered_data["NO2_pred"]) & ~np.isinf(filtered_data["NO2_sd"])]
            if not filtered_data.empty:
                # Prediction heatmap
                pred_plot = px.density_heatmap(
                    filtered_data,
                    x="Easting",
                    y="Northing",
                    z="NO2_pred",
                    facet_col="Year",
                    facet_col_wrap=3,
                    color_continuous_scale="Viridis",
                    title="Predicted NO₂ Concentration in London (2014–2019)",
                    labels={"NO2_pred": "NO₂ (µg/m³)", "Easting": "Easting", "Northing": "Northing"},
                    height=600
                )
                pred_plot.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))  # Equal aspect ratio
                pred_plot = pred_plot.to_json()

                # Uncertainty heatmap
                uncertainty_plot = px.density_heatmap(
                    filtered_data,
                    x="Easting",
                    y="Northing",
                    z="NO2_sd",
                    facet_col="Year",
                    facet_col_wrap=3,
                    color_continuous_scale="Magma",  # Matches viridis option="A"
                    title="Prediction Uncertainty (Standard Deviation) of NO₂ (2014–2019)",
                    labels={"NO2_sd": "Uncertainty (SD µg/m³)", "Easting": "Easting", "Northing": "Northing"},
                    height=600
                )
                uncertainty_plot.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))  # Equal aspect ratio
                uncertainty_plot = uncertainty_plot.to_json()
            else:
                error_message += f"No valid data for selected years {selected_years} in heatmaps."
        except Exception as e:
            error_message += f"Error rendering heatmaps: {str(e)}"

    if data["sitedata"] is not None:
        try:
            sitedata_clean = data["sitedata"].dropna(subset=["NO2", "NDVI"])
            sitedata_clean = sitedata_clean[~np.isinf(sitedata_clean["NO2"]) & ~np.isinf(sitedata_clean["NDVI"])]
            if not sitedata_clean.empty:
                scatter_plot = px.scatter(
                    sitedata_clean,
                    x="NO2",
                    y="NDVI",
                    trendline="ols",
                    title="Scatterplot NO2 vs NDVI",
                    labels={"NO2": "NO₂ (µg/m³)", "NDVI": "NDVI"}
                ).to_json()
            else:
                error_message += "No valid data for scatterplot."
        except Exception as e:
            error_message += f"Error rendering scatterplot: {str(e)}"

        try:
            time_series = data["sitedata"].groupby("Year")["NO2"].mean().reset_index()
            time_series = time_series.dropna()
            time_series = time_series[~np.isinf(time_series["NO2"])]
            if not time_series.empty:
                time_series_plot = px.line(
                    time_series,
                    x="Year",
                    y="NO2",
                    title="Evolution of Average NO2 per Year",
                    labels={"NO2": "NO₂ (µg/m³)", "Year": "Year"}
                ).to_json()
            else:
                error_message += "No valid data for time series plot."
        except Exception as e:
            error_message += f"Error rendering time series: {str(e)}"

        try:
            sites = data["sitedata"][["siteid", "Easting", "Northing"]].drop_duplicates()
            sites_gdf = gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites["Easting"], sites["Northing"]), crs=crs_bng)
            sites_gdf = sites_gdf.to_crs(crs_wgs84)
            m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
            marker_cluster = MarkerCluster().add_to(m)
            for idx, row in sites_gdf.iterrows():
                folium.Marker([row.geometry.y, row.geometry.x], popup=row["siteid"]).add_to(marker_cluster)
            map_html = m._repr_html_()
        except Exception as e:
            error_message += f"Error rendering map: {str(e)}"

    if error_message:
        debug_message += f"\nErrors: {error_message}"

    return render_template("index.html", debug_message=debug_message, years_available=years_available,
                           cv_summary=results["cv_summary"], pred_plot=pred_plot, uncertainty_plot=uncertainty_plot,
                           scatter_plot=scatter_plot, time_series_plot=time_series_plot, map_html=map_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)