import io, os, datetime as dt, requests, subprocess, json, uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import plotly.express as px
import plotly.graph_objects as go
import meteo  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS", "POST"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- Ensure static directory exists before mounting ---
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# --- Mount static directory to serve analysis results ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Analysis Request Model ---
class AnalyzeRequest(BaseModel):
    start_date: str
    end_date: str
    geometry: dict
    index_type: str = "BOTH"

# --- Analyze Endpoint ---
@app.post("/analyze")
def analyze_burn(req: AnalyzeRequest):
    # 1. Setup unique job ID and output directory
    job_id = str(uuid.uuid4())
    out_dir = os.path.join(STATIC_DIR, "analysis", job_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Save AOI to a temporary JSON file
    aoi_path = os.path.join(out_dir, "aoi.json")
    feature = {"type": "Feature", "properties": {}, "geometry": req.geometry}
    fc = {"type": "FeatureCollection", "features": [feature]}
    
    with open(aoi_path, "w") as f:
        json.dump(fc, f)
        
    # 3. Locate the script
    script_path = os.path.join(BASE_DIR, "agro.py")
    if not os.path.exists(script_path):
        # Fallback to older name if agro.py is missing
        script_path = os.path.join(BASE_DIR, "burnfinder_agri_complete.py")
        if not os.path.exists(script_path):
            return JSONResponse(status_code=500, content={"error": "Analysis script not found on server."})

    # 4. Construct Command
    # python agro.py --start X --end Y --aoi Z --out O --index {req.index_type}
    cmd = [
        "python", script_path,
        "--start", req.start_date,
        "--end", req.end_date,
        "--aoi", aoi_path,
        "--out", out_dir,
        "--index", req.index_type
    ]
    
    # 5. Run Script (Blocking execution for simplicity)
    try:
        # Capture BOTH stdout and stderr so we can debug "No Data" or API errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Script Output:", result.stdout)
        
        # --- NEW: Extract AI Analysis Text to send to Frontend ---
        report_path = os.path.join(out_dir, "burn_dates.json")
        ai_text = "Analysis complete, but AI report text not found in JSON."
        
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    data = json.load(f)
                    # Safely get the analysis string
                    ai_text = data.get("ai_analysis", "AI analysis field missing in JSON result.")
            except Exception as e:
                ai_text = f"Error reading result file: {str(e)}"

        report_url = f"/static/analysis/{job_id}/burn_dates.json"
        
        # Return success AND the text
        return {
            "status": "success", 
            "job_id": job_id, 
            "report_url": report_url,
            "ai_analysis": ai_text 
        }

    except subprocess.CalledProcessError as e:
        # Combine Output so we see the REAL error (e.g. from stderr)
        error_message = f"STDOUT: {e.stdout}\n\nSTDERR: {e.stderr}"
        print("Script Failed:", error_message)
        return JSONResponse(status_code=500, content={"status": "error", "message": error_message})


@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.get("/windmills_turkey.geojson")
def serve_windmills():
    return FileResponse("windmills_turkey.geojson", media_type="application/json")

@app.get("/solar.geojson")
def serve_solar():
    return FileResponse("solar.geojson", media_type="application/json")

@app.get("/tr-cities.json")
def serve_cities():
    return FileResponse("tr-cities.json", media_type="application/json")

@app.get("/wind-power.png")
def serve_wind_icon():
    return FileResponse("wind-power.png", media_type="image/png")

@app.get("/weather.html")
def serve_weather_page():
    return FileResponse("weather.html", media_type="text/html")


# 3) Configuration & common loader
FILE_PATH = "lightening.xlsx"
CENTER_LAT = 36.411611
CENTER_LNG = 36.112389
R_EARTH_KM = 6371.0

def load_and_process():
    df = pd.read_excel(FILE_PATH)
    df = df[df["p_type"] == 1]
    df["abs_current"] = df["current"].abs() / 1000.0

    # Convert to radians
    df["lat_rad"] = np.radians(df["lat"])
    df["lng_rad"] = np.radians(df["lng"])
    clat = np.radians(CENTER_LAT)
    clng = np.radians(CENTER_LNG)

    # Equirectangular approximation
    df["dx"] = (df["lng_rad"] - clng) * R_EARTH_KM * np.cos(clat)
    df["dy"] = (df["lat_rad"] - clat) * R_EARTH_KM
    df["distance_km"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    return df

# 4) Static Matplotlib scatter
@app.get("/scatter")
def scatter_plot():
    df = load_and_process()
    def cat(i):
        if i <= 8: return "Low (≤8 kA)"
        elif i <= 20: return "Medium (8–20 kA)"
        else: return "High (>20 kA)"
    df["category"] = df["abs_current"].apply(cat)

    fig, ax = plt.subplots(figsize=(6,6))
    colors = {"Low (≤8 kA)":"blue","Medium (8–20 kA)":"orange","High (>20 kA)":"red"}
    for label, color in colors.items():
        sub = df[df["category"] == label]
        ax.scatter(sub["dx"], sub["dy"], label=label, s=10, c=color, alpha=0.6)

    ax.plot(0,0,"*",c="black",ms=12,label="Center")
    for r in (5,10,20):
        circ = plt.Circle((0,0), r, color="black", ls="--", fill=False)
        ax.add_artist(circ)

    ax.set_title("Lightning Strike Distribution")
    ax.set_xlabel("East–West Distance (km)")
    ax.set_ylabel("North–South Distance (km)")
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# 5) Static Matplotlib histogram
@app.get("/bar")
def distance_histogram():
    df = load_and_process()
    bins = [0,5,10,20,50]
    labels = ["0–5 km","5–10 km","10–20 km","20–50 km"]
    df["bin"] = pd.cut(df["distance_km"], bins=bins, labels=labels, right=False)
    counts = df["bin"].value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(6,5))
    bars = ax.bar(counts.index, counts.values, color="purple")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from Center")
    ax.set_ylabel("Number of Strikes (log scale)")
    ax.set_title("Strike Count by Distance")
    ax.grid(True, which="both", axis="y", ls="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0,5), textcoords="offset points",
                    ha="center", va="bottom")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# 6) Interactive Plotly scatter only
@app.get("/scatter_int")
def scatter_interactive():
    df = load_and_process()
    df["CG_amp"]     = df["current"] / 1000.0
    df["abs_current"]= df["CG_amp"].abs()

    fig = px.scatter(
        df,
        x="dx", y="dy",
        color="abs_current",
        color_continuous_scale="Turbo",
        hover_data={"CG_amp": True, "distance_km": True},
        labels={
            "dx":          "East–West (km)",
            "dy":          "North–South (km)",
            "abs_current":"│CG│ (kA)",
            "CG_amp":     "Signed CG (kA)"
        },
        title="Interactive Lightning Scatter (CG Amplitudes)"
    )

    # concentric circles
    for r in (5, 10, 20):
        fig.add_shape(
            type="circle",
            x0=-r, y0=-r,
            x1=r,  y1=r,
            line=dict(color="black", dash="dash"),
        )

    # center star
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(symbol="star", size=12, color="black"),
            name="Center",
        )
    )

    fig.update_layout(
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1
        ),
        margin=dict(r=120)
    )
    fig.update_coloraxes(colorbar=dict(x=1.05))

    return HTMLResponse(fig.to_html(full_html=True, include_plotlyjs="cdn"))


# ----------------- Weather JSON -----------------
@app.get("/weather.json")
def weather_json(
    lat: float = Query(...),
    lon: float = Query(...),
    days: int = 7,
    rated_power_mw: float | None = None,
    num_turbines: int = 1,
    system_size_kw: float | None = None,
):
    df = meteo.fetch_hourly(
        lat, lon, days=days, timezone="UTC",
        rated_power_mw=rated_power_mw,
        num_turbines=num_turbines,
        system_size_kw=system_size_kw,
        performance_ratio=0.80
    )
    daily = df.attrs.get("daily").reset_index().rename(columns={"date": "day"})
    summary = df.attrs.get("summary", {})
    return {
        "hourly": df.reset_index().to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
        "summary": summary
    }

# ----------------- PDF Helpers -----------------
def _capacity_factor_from_wind(v, cut_in=3.0, rated=12.0, cut_out=25.0):
    if v < cut_in or v >= cut_out:
        return 0.0
    if v >= rated:
        return 1.0
    x = (v - cut_in) / (rated - cut_in)
    return max(0.0, min(1.0, x**3))

def _fetch_daily_hourly(lat, lon, start_iso, end_iso, timezone="auto"):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_iso}&end_date={end_iso}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,"
        "windspeed_10m_max,winddirection_10m_dominant,uv_index_max,shortwave_radiation_sum"
        "&hourly=wind_speed_10m,uv_index,shortwave_radiation"
        f"&timezone={timezone}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def _enrich_daily_for_report(data, rated_power_mw, cut_in, rated, cut_out, panel_area_m2, panel_efficiency):
    days = data["daily"]["time"]
    hourly_t = data.get("hourly", {}).get("time", []) or []
    hourly_w = data.get("hourly", {}).get("wind_speed_10m", []) or []
    hourly_uv = data.get("hourly", {}).get("uv_index", []) or []

    buckets_wind, buckets_uv = {}, {}
    for t, v in zip(hourly_t, hourly_w):
        buckets_wind.setdefault(t.split("T")[0], []).append(float(v))
    for t, v in zip(hourly_t, hourly_uv):
        buckets_uv.setdefault(t.split("T")[0], []).append(float(v))

    ghi_kwh_m2 = [(mj or 0)/3.6 for mj in data["daily"].get("shortwave_radiation_sum", [])]

    rows = []
    for i, d in enumerate(days):
        uv_avg = (sum(buckets_uv[d])/len(buckets_uv[d])) if d in buckets_uv else data["daily"].get("uv_index_max",[None])[i]
        wind_mwh = None
        if d in buckets_wind and rated_power_mw > 0:
            cfs = [_capacity_factor_from_wind(v, cut_in, rated, cut_out) for v in buckets_wind[d]]
            wind_mwh = sum(cfs) * rated_power_mw
        solar_kwh = None
        if panel_area_m2 > 0 and panel_efficiency > 0:
            ghi = ghi_kwh_m2[i] if i < len(ghi_kwh_m2) else 0
            solar_kwh = ghi * panel_area_m2 * panel_efficiency
        rows.append({
            "date": d,
            "temp_max": data["daily"]["temperature_2m_max"][i],
            "temp_min": data["daily"]["temperature_2m_min"][i],
            "precip_prob": data["daily"]["precipitation_probability_max"][i],
            "precip_sum": data["daily"]["precipitation_sum"][i],
            "wind_speed_max": data["daily"]["windspeed_10m_max"][i],
            "wind_dir_dom": data["daily"]["winddirection_10m_dominant"][i],
            "uv_index_avg": uv_avg,
            "wind_energy_per_turbine_mwh": wind_mwh,
            "solar_energy_per_panel_kwh": solar_kwh,
        })
    return pd.DataFrame(rows)

def _try_read_image(path):
    try:
        return mpimg.imread(path)
    except Exception:
        return None

def _build_pdf_report(site_type, lat, lon, df, rated_power_mw, panel_area_m2, panel_efficiency, props=None):
    df_fmt = df.copy().round({
        "temp_max":1,"temp_min":1,"precip_prob":0,"precip_sum":1,
        "wind_speed_max":1,"uv_index_avg":2,
        "wind_energy_per_turbine_mwh":2,"solar_energy_per_panel_kwh":2
    })
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---------- Page 1: Cover ----------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")

        bg = _try_read_image(os.path.join(STATIC_DIR, "cover_background.png"))
        if bg is not None:
            ax.imshow(bg, extent=[0, 1, 0, 1], aspect="auto", zorder=-1)

        logo = _try_read_image(os.path.join(STATIC_DIR, "logo.png"))
        if logo is not None:
            ax_logo = fig.add_axes([0.42, 0.88, 0.16, 0.08])  
            ax_logo.axis("off")
            ax_logo.imshow(logo)

        ax.text(0.5, 0.75, "Weather & Energy Report",
                ha="center", fontsize=22, weight="bold")
        ax.text(0.5, 0.68, f"Site type: {site_type.capitalize()}",
                ha="center", fontsize=14)
        ax.text(0.5, 0.64, f"Coords: {lat:.4f}, {lon:.4f}",
                ha="center", fontsize=12)
        ax.text(0.5, 0.60, f"Period: {df['date'].iloc[0]} → {df['date'].iloc[-1]}",
                ha="center", fontsize=12)

        details = (
            f"Turbine rated power (per unit): {rated_power_mw} MW\n"
            f"Panel (per unit): area {panel_area_m2} m² • efficiency {panel_efficiency*100:.0f}%"
        )
        ax.text(0.5, 0.52, details, ha="center", fontsize=11)

        pdf.savefig(fig); plt.close(fig)


        if props:
            clean_props = {k:v for k,v in props.items() if v not in (None,"","null")}
            fig,ax = plt.subplots(figsize=(8.27,11.69)); ax.axis("off")
            ax.text(0.5,0.95,"Site Information",ha="center",fontsize=16,weight="bold")
            y=0.90
            for k,v in clean_props.items():
                ax.text(0.1,y,f"{k}: {v}",fontsize=12,va="top"); y-=0.045
            pdf.savefig(fig); plt.close(fig)


        fig,ax=plt.subplots(figsize=(11.69,8.27)); ax.axis("off")
        table_df=df_fmt[["date","temp_max","temp_min","precip_prob","precip_sum",
                         "wind_speed_max","wind_dir_dom","uv_index_avg",
                         "wind_energy_per_turbine_mwh","solar_energy_per_panel_kwh"]].copy()
        table_df.columns=["Date","Tmax (°C)","Tmin (°C)","PrecipProb","Precip","WindMax","WindDir","UV",
                          "Wind (MWh)","Solar (kWh)"]
        tbl=ax.table(cellText=table_df.values,colLabels=table_df.columns,loc="center",cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.4)
        pdf.savefig(fig); plt.close(fig)


        x=np.arange(len(df_fmt))
        fig,ax=plt.subplots(figsize=(11.69,8.27))
        ax.plot(x,df_fmt["temp_max"],label="Tmax (°C)")
        ax.plot(x,df_fmt["temp_min"],label="Tmin (°C)")
        ax.plot(x,df_fmt["wind_speed_max"],label="WindMax (km/h)")
        ax.set_xticks(x,df_fmt["date"],rotation=45,ha="right")
        ax.legend(); ax.set_title("Temperature & Wind Speed")
        pdf.savefig(fig); plt.close(fig)


        fig,ax=plt.subplots(figsize=(11.69,8.27))
        ax.bar(x-0.2,df_fmt["precip_prob"],width=0.4,label="PrecipProb")
        ax.bar(x+0.2,df_fmt["precip_sum"],width=0.4,label="Precip")
        ax.set_xticks(x,df_fmt["date"],rotation=45,ha="right")
        ax.legend(); ax.set_title("Precipitation")
        pdf.savefig(fig); plt.close(fig)


        fig,ax=plt.subplots(figsize=(11.69,8.27))
        ax.plot(x,df_fmt["wind_energy_per_turbine_mwh"],marker="o",label="Wind/Turbine MWh")
        ax.set_xticks(x,df_fmt["date"],rotation=45,ha="right")
        ax.legend(); ax.set_title("Wind Energy per Turbine")
        pdf.savefig(fig); plt.close(fig)


        fig,ax=plt.subplots(figsize=(11.69,8.27))
        ax.plot(x,df_fmt["solar_energy_per_panel_kwh"],marker="s",color="orange",label="Solar/Panel kWh")
        ax.set_xticks(x,df_fmt["date"],rotation=45,ha="right")
        ax.legend(); ax.set_title("Solar Energy per Panel")
        pdf.savefig(fig); plt.close(fig)


    return buf.getvalue()

# ----------------- Report Route -----------------
@app.get("/report.pdf")
def report_pdf(
    lat: float, lon: float, days: int = 7,
    site_type: str = "wind",
    rated_power_mw: float = 3.6,
    cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0,
    panel_area_m2: float = 2.0, panel_efficiency: float = 0.20,
    # Wind props
    name: str | None = None, status: str | None = None, year: str | None = None,
    # Solar props
    capacity: str | None = None, project: str | None = None, operator: str | None = None,
    province: str | None = None, method: str | None = None, plant_source: str | None = None, power: str | None = None,
):
    end=dt.date.today(); start=end-dt.timedelta(days=days-1)
    data=_fetch_daily_hourly(lat,lon,start.isoformat(),end.isoformat())
    df=_enrich_daily_for_report(data,rated_power_mw,cut_in,rated,cut_out,panel_area_m2,panel_efficiency)

    props={
        "Name": name,"Status": status,"Year": year,
        "Project": project,"Operator": operator,"Capacity": capacity,
        "Province": province,"Method": method,"Plant Source": plant_source,"Power": power,
        "Latitude": f"{lat:.4f}","Longitude": f"{lon:.4f}"
    }

    pdf_bytes=_build_pdf_report(site_type,lat,lon,df,rated_power_mw,panel_area_m2,panel_efficiency,props)
    return StreamingResponse(io.BytesIO(pdf_bytes),media_type="application/pdf",
                             headers={"Content-Disposition":"attachment; filename=weather_report.pdf"})