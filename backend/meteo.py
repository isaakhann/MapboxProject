# meteo.py
from __future__ import annotations
from datetime import date, timedelta
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

_session = requests_cache.CachedSession(".cache", expire_after=3600)
_session = retry(_session, retries=2, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_session)

def _capacity_factor_from_wind(
    v_ms: float,
    cut_in: float = 3.0,       # m/s
    rated: float = 12.0,       # m/s
    cut_out: float = 25.0      # m/s
) -> float:
    if v_ms < cut_in or v_ms >= cut_out:
        return 0.0
    if v_ms >= rated:
        return 1.0
    x = (v_ms - cut_in) / (rated - cut_in)
    return max(0.0, min(1.0, x ** 3))

def fetch_historical_hourly(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
    *,
    rated_power_mw: float | None = None,     
    num_turbines: int = 1,
    cut_in_ms: float = 3.0,
    rated_ms: float = 12.0,
    cut_out_ms: float = 25.0,
    air_density: float = 1.225,             
    system_size_kw: float | None = 1.0,      
    performance_ratio: float = 0.80,         
    panel_area_m2: float | None = None,      
    panel_efficiency: float = 0.20,         
) -> pd.DataFrame:
    """
    Returns an hourly DataFrame with:
      - temperature_2m, wind_speed_10m, wind_direction_10m, precipitation_probability
      - uv_index
      - shortwave_radiation (W/m²) and derived hourly solar energy
      - wind_energy_mwh (per plant if rated_power_mw & num_turbines set; else 0)
      - solar_energy_kwh (either per kW installed using PR, or per m² * efficiency, or both)
    Also attaches daily sums/means in df.attrs['daily'] and period summaries in df.attrs['summary'].
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "hourly": ",".join([
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation_probability",
            "uv_index",
            "shortwave_radiation"
        ])
    }

    resp = _client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    if len(resp) == 0:
        raise RuntimeError("Open-Meteo returned no results.")

    ds = resp[0]
    hourly = ds.Hourly()

    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(timezone),
        periods=hourly.Variables(0).ValuesAsNumpy().size,
        freq=pd.Timedelta(seconds=hourly.Interval())
    )

    def v(name: str):
        return hourly.Variables([v.Name() for v in hourly.VariablesList()].index(name)).ValuesAsNumpy()

    df = pd.DataFrame(index=time_index)
    df.index.name = "date"

    df["temperature_2m"] = v("temperature_2m")
    df["wind_speed_10m"] = v("wind_speed_10m")
    df["wind_direction_10m"] = v("wind_direction_10m")
    df["precipitation_probability"] = v("precipitation_probability")
    df["uv_index"] = v("uv_index")

    df["shortwave_radiation"] = v("shortwave_radiation")

    plant_rated_mw = (rated_power_mw or 0.0) * max(1, int(num_turbines))
    if plant_rated_mw <= 0:
        df["wind_energy_mwh"] = 0.0
    else:
        cf = df["wind_speed_10m"].apply(lambda x: _capacity_factor_from_wind(x, cut_in_ms, rated_ms, cut_out_ms))
        df["wind_energy_mwh"] = plant_rated_mw * cf

    df["ghi_kwh_m2"] = df["shortwave_radiation"] / 1000.0

    solar_kwh = 0.0
    if system_size_kw is not None:
        solar_kwh = df["ghi_kwh_m2"] * performance_ratio * float(system_size_kw)

    if panel_area_m2 is not None:
        solar_kwh += df["ghi_kwh_m2"] * float(panel_area_m2) * float(panel_efficiency)

    df["solar_energy_kwh"] = solar_kwh

    daily = pd.DataFrame({
        "wind_energy_mwh": df["wind_energy_mwh"].resample("1D").sum(min_count=1),
        "solar_energy_kwh": df["solar_energy_kwh"].resample("1D").sum(min_count=1),
        "uv_index_avg":     df["uv_index"].resample("1D").mean(),
        "wind_speed_avg":   df["wind_speed_10m"].resample("1D").mean(),
        "ghi_kwh_m2":       df["ghi_kwh_m2"].resample("1D").sum(min_count=1),
    })
    df.attrs["daily"] = daily

    summary = {
        "period_hours": int(len(df)),
        "wind_energy_mwh_total": float(df["wind_energy_mwh"].sum()),
        "solar_energy_kwh_total": float(df["solar_energy_kwh"].sum()),
        "uv_index_mean": float(df["uv_index"].mean()),
        "wind_speed_mean_ms": float(df["wind_speed_10m"].mean()),
        "ghi_kwh_m2_total": float(df["ghi_kwh_m2"].sum()),
    }
    df.attrs["summary"] = summary

    return df


def fetch_hourly(
    latitude: float,
    longitude: float,
    days: int = 7,
    timezone: str = "UTC",
    **kwargs
) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=days)
    return fetch_historical_hourly(
        latitude, longitude,
        start.isoformat(), end.isoformat(),
        timezone=timezone,
        **kwargs
    )
