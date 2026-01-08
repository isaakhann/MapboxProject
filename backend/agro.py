#!/usr/bin/env python3
"""
burnfinder_agri_complete.py — Detect likely burn dates and calculate agricultural/water indices
over an AOI using Sentinel-2 L2A.

This version:
- Uses paging in STAC search (prevents timeouts).
- Computes: BAI, BAIS2, NDVI, NDWI, NDMI, NDRE (+ NBR for gating).
- Exports PNG quicklooks and optional GeoTIFF rasters.
- Generates post-run: event_summary.md + event_summary.pdf (from burn_dates.json)
  including:
    • burned or not + confidence
    • event anchor date + how chosen
    • step changes for BAIS2/BAI/NBR/NDVI/NDMI/NDWI
    • optional tillage inference

Important: Index math, masking, detection logic are NOT changed.
Only visualization for BAIS2 PNG uses the requested blue→yellow→red scheme.
"""

import os, io, json, sys, warnings, csv
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from rasterio.enums import Resampling

import geopandas as gpd
from shapely.ops import unary_union

from pystac_client import Client
import planetary_computer as pc
import stackstac

from tqdm import tqdm
import requests
import click
from PIL import Image
from scipy.ndimage import uniform_filter1d
from pyproj import Transformer

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# IO helpers
# ---------------------------

def ensure_dir(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def _looks_like_wkt(s: str) -> bool:
    """Check if a string resembles Well-Known Text (WKT)."""
    s2 = s.strip().upper()
    return s2.startswith(("POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON",
                          "MULTILINESTRING", "MULTIPOINT", "GEOMETRYCOLLECTION"))

def _looks_like_geojson(s: str) -> bool:
    """Check if a string resembles GeoJSON (starts with '{' or '[')."""
    s2 = s.strip()
    return s2.startswith("{") or s2.startswith("[")

def load_aoi(path_or_text: str) -> gpd.GeoDataFrame:
    """
    Load an Area of Interest (AOI) from a file path, GeoJSON string, or WKT.
    Converts all inputs to EPSG:4326.
    """
    try:
        if os.path.exists(path_or_text):
            gdf = gpd.read_file(path_or_text)
            if gdf.empty:
                raise ValueError("AOI file is empty.")
            return gdf.to_crs(4326)
        if _looks_like_geojson(path_or_text):
            obj = json.loads(path_or_text)
            if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
                gdf = gpd.GeoDataFrame.from_features(obj["features"], crs="EPSG:4326")
            else:
                from shapely.geometry import shape
                geom = shape(obj["geometry"] if isinstance(obj, dict) and "geometry" in obj else obj)
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            return gdf.to_crs(4326)
        if _looks_like_wkt(path_or_text):
            from shapely import wkt
            geom = wkt.loads(path_or_text)
            return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        if any(path_or_text.lower().endswith(ext) for ext in [".geojson", ".json", ".gpkg", ".shp"]):
            raise FileNotFoundError(f"AOI path not found: {path_or_text}")
        raise ValueError("AOI must be a valid file path, raw GeoJSON string, or WKT.")
    except Exception as e:
        raise ValueError(f"Failed to read AOI: {e}")

# ---------------------------
# STAC search (Planetary Computer)
# ---------------------------

def search_sentinel2(aoi_gdf: gpd.GeoDataFrame, start: str, end: str, max_items: int = 300) -> List[Dict[str, Any]]:
    """
    Search the Planetary Computer STAC endpoint for Sentinel-2 L2A items using paging
    to prevent search timeouts.
    """
    client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    bbox = list(aoi_gdf.total_bounds)
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": 100}},  # relaxed filter
        max_items=max_items,
        sortby=[{"field": "datetime", "direction": "asc"}],
    )

    items: List[Dict[str, Any]] = []
    for i in tqdm(search.items(), desc="Fetching STAC Items (Paging)"):
        items.append(pc.sign(i).to_dict())
        if len(items) >= max_items:
            break
    return items

# ---------------------------
# Asset loading / clipping
# ---------------------------

def _asset_scale(item_dict: dict, asset_name: str) -> float:
    """Determine scale factor for Sentinel-2 reflectance assets."""
    a = item_dict.get("assets", {}).get(asset_name, {})
    rb = a.get("raster:bands") or a.get("raster_bands")
    if rb and isinstance(rb, list) and rb and isinstance(rb[0], dict):
        sc = rb[0].get("scale")
        if isinstance(sc, (int, float)):
            return float(sc)
    sc = a.get("scale")
    if isinstance(sc, (int, float)):
        return float(sc)
    if asset_name in {"B01","B02","B03","B04","B05","B06","B07","B8A","B08","B11","B12"}:
        return 0.0001
    return 1.0

def _reproject_bounds(bounds: tuple, src_epsg: int, dst_epsg: int) -> tuple:
    """Reproject bounding box coordinates from source to destination CRS."""
    minx, miny, maxx, maxy = bounds
    tf = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    x1, y1 = tf.transform(minx, miny)
    x2, y2 = tf.transform(maxx, maxy)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def _load_clip_fallback(item_dict: Dict[str, Any], aoi_gdf_4326: gpd.GeoDataFrame, available_assets: List[str]) -> xr.Dataset:
    """
    Stabilized fallback loading, defining the template from the first successfully loaded asset.
    Continues even if some bands fail.
    """
    proj_epsg = item_dict.get("properties", {}).get("proj:epsg")

    if proj_epsg is not None:
        epsg_arg = int(proj_epsg)
        bounds_arg = list(aoi_gdf_4326.to_crs(epsg_arg).total_bounds)
    else:
        epsg_arg = 4326
        bounds_arg = list(aoi_gdf_4326.total_bounds)

    data_vars: Dict[str, Tuple[Tuple[str, str], np.ndarray]] = {}
    template = None
    success = False

    for b in available_assets:
        try:
            href = item_dict["assets"][b]["href"]
            arr = rxr.open_rasterio(href, masked=True)

            if arr.rio.crs is None and proj_epsg is not None:
                arr = arr.rio.write_crs(epsg_arg)
            elif arr.rio.crs is None:
                raise RuntimeError(f"Asset {b} has no discernible CRS.")

            bounds_local = _reproject_bounds(bounds_arg, epsg_arg, arr.rio.crs.to_epsg())
            arr = arr.rio.clip_box(*bounds_local)

            if template is None:
                template = arr.rio.reproject(arr.rio.crs, resolution=10)

            resampling_method = rio.enums.Resampling.nearest if b == "SCL" else rio.enums.Resampling.bilinear
            arr = arr.rio.reproject_match(template, resampling=resampling_method)

            data = arr.squeeze().values
            if b != "SCL":
                scale = _asset_scale(item_dict, b)
                data = data.astype("float32") * scale

            data_vars[b] = (("y", "x"), data)
            success = True
        except Exception:
            continue

    if not success or template is None:
        raise RuntimeError("Fallback loading failed for all requested assets.")

    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs(template.rio.crs)
    ds = ds.rio.write_transform(template.rio.transform())
    return ds

def load_clip_item(item_dict: Dict[str, Any], aoi_gdf_4326: gpd.GeoDataFrame) -> xr.Dataset:
    """Load a Sentinel-2 item into an xarray Dataset, clipped to the AOI."""
    assets = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","SCL"]
    available_assets = [a for a in assets if a in item_dict.get("assets", {})]

    try:
        stack = stackstac.stack(
            [item_dict],
            assets=available_assets,
            bounds=aoi_gdf_4326.total_bounds,
            epsg=4326,
            resolution=10,
            resampling=Resampling.bilinear,
            chunksize=2048,
        )
        stack = stack.squeeze("time", drop=True)
        ds = stack.to_dataset(dim="band")
        return ds
    except Exception:
        return _load_clip_fallback(item_dict, aoi_gdf_4326, available_assets)

# ---------------------------
# Masking
# ---------------------------

def build_clear_mask(ds: xr.Dataset) -> xr.DataArray:
    """Build a clear mask using Sentinel-2 SCL classes 4,5,6,7."""
    scl = ds["SCL"]
    clear = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)
    return clear

# ---------------------------
# Indices
# ---------------------------

def compute_NBR(nir: xr.DataArray, swir2: xr.DataArray) -> xr.DataArray:
    """Normalized Burn Ratio."""
    return (nir - swir2) / (nir + swir2)

def compute_BAI(ds: xr.Dataset) -> xr.DataArray:
    """Burned Area Index (BAI). Applies water masking using SCL."""
    red = ds["B04"]
    nir = ds["B08"]
    scl = ds["SCL"]
    water_mask = ~(scl == 6)
    bai = 1.0 / (((0.1 - red) ** 2) + ((0.06 - nir) ** 2))
    return bai.where(water_mask)

def compute_BAIS2(ds: xr.Dataset) -> xr.DataArray:
    """BAIS2 burn index."""
    b04 = ds["B04"]
    b06 = ds["B06"]
    b07 = ds["B07"]
    b8a = ds["B8A"]
    b12 = ds["B12"]
    return (1 - np.sqrt((b06 * b07 * b8a) / b04)) * (((b12 - b8a) / np.sqrt(b12 + b8a)) + 1)

def compute_NDVI(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
    return (nir - red) / (nir + red)

def compute_NDWI(green: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
    return (green - nir) / (green + nir)

def compute_NDMI(nir: xr.DataArray, swir1: xr.DataArray) -> xr.DataArray:
    return (nir - swir1) / (nir + swir1)

def compute_NDRE(nir: xr.DataArray, red_edge: xr.DataArray) -> xr.DataArray:
    return (nir - red_edge) / (nir + red_edge)

# ---------------------------
# Quicklooks (existing)
# ---------------------------

def _nanpercentile(a: np.ndarray, q: float) -> float:
    try:
        return float(np.nanpercentile(a, q))
    except Exception:
        return float("nan")

def _normalize_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = (arr - vmin) / (vmax - vmin + 1e-9)
    x = np.clip(x, 0, 1)
    return (x * 255).astype("uint8")

def _resize_for_viz(img: Image.Image, viz_scale: int = 1, viz_min_size: int = 1500) -> Image.Image:
    if viz_scale and viz_scale != 1:
        img = img.resize((img.size[0] * viz_scale, img.size[1] * viz_scale), resample=Image.BICUBIC)

    w, h = img.size
    short = min(w, h)
    if viz_min_size and short < viz_min_size:
        factor = viz_min_size / float(short)
        img = img.resize((int(w * factor), int(h * factor)), resample=Image.BICUBIC)
    return img

def save_png_gray(arr: np.ndarray, path: str, vmin: Optional[float] = None, vmax: Optional[float] = None,
                  viz_scale: int = 1, viz_min_size: int = 1500) -> None:
    a = np.array(arr, dtype="float32")
    if vmin is None: vmin = float(np.nanpercentile(a, 1))
    if vmax is None: vmax = float(np.nanpercentile(a, 99))
    img = Image.fromarray(_normalize_to_uint8(a, vmin, vmax), mode="L").convert("RGBA")
    img = _resize_for_viz(img, viz_scale=viz_scale, viz_min_size=viz_min_size)
    img.save(path)

def save_png_agri_color(arr: np.ndarray, path: str, vmin: float, vmax: float,
                        viz_scale: int = 1, viz_min_size: int = 1500) -> None:
    # Keeps existing “greenish” renderer (unchanged logic)
    a = np.array(arr, dtype="float32")
    g = _normalize_to_uint8(a, vmin, vmax)
    rgb = np.zeros((g.shape[0], g.shape[1], 3), dtype="uint8")
    rgb[..., 1] = g
    rgb[..., 0] = (g * 0.25).astype("uint8")
    rgb[..., 2] = (g * 0.1).astype("uint8")
    img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    img = _resize_for_viz(img, viz_scale=viz_scale, viz_min_size=viz_min_size)
    img.save(path)

def render_truecolor(ds: xr.Dataset) -> np.ndarray:
    r = ds["B04"].values
    g = ds["B03"].values
    b = ds["B02"].values
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb ** (1/2.2) * 255).astype("uint8")
    return rgb

def save_png_rgb(rgb: np.ndarray, path: str, viz_scale: int = 1, viz_min_size: int = 1500) -> None:
    img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    img = _resize_for_viz(img, viz_scale=viz_scale, viz_min_size=viz_min_size)
    img.save(path)

# ---------------------------
# BAIS2 quicklook color scheme (NEW, visualization-only)
# ---------------------------

def _upscale(img: Image.Image, viz_scale: int, viz_min_size: int) -> Image.Image:
    w, h = img.size
    scale = int(viz_scale) if viz_scale else 1
    if viz_min_size:
        scale = max(scale, int(np.ceil(viz_min_size / max(1, min(w, h)))))
    if scale > 1:
        return img.resize((w * scale, h * scale), resample=Image.BICUBIC)
    return img

def save_png_bais2_color(
    arr: np.ndarray,
    path: str,
    viz_scale: int = 1,
    viz_min_size: int = 1500,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    BAIS2 quicklook in the requested style: blue/cyan background -> yellow -> red hotspots.
    Visualization only; analysis logic unchanged.
    """
    zero_break = 0.5
    low_color  = np.array([0/255,   0/255, 255/255], dtype=np.float32)   # blue
    mid_color  = np.array([250/255, 255/255, 10/255], dtype=np.float32)  # yellow
    high_color = np.array([255/255,  20/255, 20/255], dtype=np.float32)  # red

    a = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(a).any():
        Image.new("RGB", (64, 64), (255, 255, 255)).save(path, optimize=True)
        return

    if vmin is None:
        vmin = float(_nanpercentile(a, 2))
    if vmax is None:
        vmax = float(_nanpercentile(a, 98))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin >= vmax):
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            Image.new("RGB", (64, 64), (255, 255, 255)).save(path, optimize=True)
            return
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin >= vmax):
            Image.new("RGB", (64, 64), (255, 255, 255)).save(path, optimize=True)
            return

    if vmin > zero_break:
        vmin = zero_break * 0.9
    if vmax < zero_break:
        vmax = zero_break * 1.1

    out = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.float32)

    mask_lo = (a <= zero_break) & np.isfinite(a)
    if mask_lo.any():
        v = a[mask_lo]
        t = np.clip((v - vmin) / (zero_break - vmin + 1e-6), 0.0, 1.0)
        out[mask_lo] = (1.0 - t)[:, None] * low_color + t[:, None] * mid_color

    mask_hi = (a > zero_break) & np.isfinite(a)
    if mask_hi.any():
        v = a[mask_hi]
        t = np.clip((v - zero_break) / (vmax - zero_break + 1e-6), 0.0, 1.0)
        out[mask_hi] = (1.0 - t)[:, None] * mid_color + t[:, None] * high_color

    out = np.clip(out, 0.0, 1.0)
    img = Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")
    img = _upscale(img, viz_scale, viz_min_size)
    img.save(path, optimize=True)

# ---------------------------
# Detection (unchanged)
# ---------------------------

def detect_burn_dates_generic(signal_mean: List[float], nbr_mean: List[float], dates: List[dt.datetime],
                             nbr_drop_gate: float = -0.03, smooth_win: int = 3) -> List[str]:
    """
    Generic change detection:
    - smooth the signal
    - find peaks that coincide with NBR drop (optional gate)
    """
    if not signal_mean or len(signal_mean) < 3:
        return []

    s = np.array(signal_mean, dtype="float32")
    s_smooth = uniform_filter1d(s, size=smooth_win, mode="nearest")

    thr = np.nanpercentile(s_smooth, 80)
    cand = []
    for i in range(1, len(s_smooth) - 1):
        if s_smooth[i] > thr and s_smooth[i] > s_smooth[i-1] and s_smooth[i] > s_smooth[i+1]:
            cand.append(i)

    out = []
    for i in cand:
        if nbr_mean and len(nbr_mean) == len(signal_mean):
            if i > 0:
                nbr_drop = float(nbr_mean[i] - nbr_mean[i-1])
                if nbr_drop > nbr_drop_gate:
                    continue
        out.append(dates[i].strftime("%Y-%m-%d"))

    return out

# ---------------------------
# Reporting
# ---------------------------

def write_index_report(outdir: str, index_name: str, formula: str,
                       stats_rows: List[Dict[str, Any]],
                       burn_candidates: Optional[List[str]] = None,
                       notes_suffix: str = "") -> Dict[str, str]:
    """Write CSV + Markdown report for an index."""
    csv_path = os.path.join(outdir, f"{index_name}_values.csv")
    md_path = os.path.join(outdir, f"{index_name}_report.md")

    if stats_rows:
        keys = list(stats_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in stats_rows:
                w.writerow(r)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {index_name} Report\n\n")
        f.write(f"**Formula:** {formula}\n\n")
        if notes_suffix:
            f.write(f"**Notes:** {notes_suffix}\n\n")
        if burn_candidates:
            f.write("## Detected candidate burn dates\n\n")
            for d in burn_candidates:
                f.write(f"- {d}\n")
            f.write("\n")

        f.write("## Summary statistics by date\n\n")
        f.write("| date | clear_frac | mean | std | p05 | p25 | p50 | p75 | p95 | max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in stats_rows:
            f.write(
                f"| {r.get('date')} | {r.get('clear_frac', float('nan')):.4f} | "
                f"{r.get('mean', float('nan')):.4f} | {r.get('std', float('nan')):.4f} | "
                f"{r.get('p05', float('nan')):.4f} | {r.get('p25', float('nan')):.4f} | "
                f"{r.get('p50', float('nan')):.4f} | {r.get('p75', float('nan')):.4f} | "
                f"{r.get('p95', float('nan')):.4f} | {r.get('max', float('nan')):.4f} |\n"
            )

    return {"csv": csv_path, "markdown": md_path}

# ---------------------------
# MODIS hotspots (optional)
# ---------------------------

def fetch_modis_hotspots(aoi_gdf: gpd.GeoDataFrame, start: str, end: str) -> set:
    """Fetches recent MODIS fire hotspots (limited to 7 days) to provide context."""
    try:
        d1, d2 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        if (d2 - d1).days > 14:
            return set()
        url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/c7/csv/MODIS_C6_1_Global_7d.csv"
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or not r.text:
            return set()

        import csv as _csv
        xmin, ymin, xmax, ymax = aoi_gdf.total_bounds
        dates = set()

        for row in _csv.DictReader(io.StringIO(r.text)):
            lat = float(row["latitude"]); lon = float(row["longitude"])
            if (xmin <= lon <= xmax) and (ymin <= lat <= ymax):
                t = dt.datetime.strptime(row["acq_date"], "%Y-%m-%d").date()
                if d1 <= t <= d2:
                    dates.add(t)

        return dates
    except Exception:
        return set()

# ---------------------------
# CLI
# ---------------------------

@click.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--aoi", required=True, help="Path to GeoJSON/GPX/GeoPackage/Shapefile OR raw GeoJSON OR WKT")
@click.option("--ask", multiple=True, default=["BurnDate"], help="What to detect (currently only BurnDate is supported)")
@click.option("--out", "outdir", required=True, help="Output directory path")
@click.option("--index", type=click.Choice(["BAI","BAIS2","BOTH","AGRI","ALL"], case_sensitive=False),
              default="BOTH", show_default=True,
              help="Which index group(s) to compute and use for detection/exports. AGRI includes NDVI, NDWI, NDMI, NDRE.")
@click.option("--use-modis", is_flag=True, help="Confirm timing with MODIS hotspots (for info only)")
@click.option("--min-clear", default=0.01, show_default=True, help="Minimum clear pixel fraction per scene (0-1) to be processed")
@click.option("--viz-scale", default=1, show_default=True, help="Upscale factor for exported PNGs (1=native clip size)")
@click.option("--viz-min-size", default=1500, show_default=True, help="Minimum short-edge size (px) for PNGs (forced upscaling)")
@click.option("--write-geotiff", is_flag=True, help="Also write GeoTIFFs (truecolor + index rasters) at native resolution")
def main(start, end, aoi, ask, outdir, index, use_modis, min_clear,
         viz_scale, viz_min_size, write_geotiff):
    """
    Main function to run the burn detection and index calculation process.
    """
    print(f"Starting BurnFinder for {start} to {end}...")
    ensure_dir(outdir)
    pics_dir = os.path.join(outdir, "pics"); ensure_dir(pics_dir)

    try:
        aoi_gdf = load_aoi(aoi)
    except ValueError as e:
        print(f"Error loading AOI: {e}"); sys.exit(1)

    aoi_union = gpd.GeoDataFrame(geometry=[unary_union(aoi_gdf.geometry)], crs=aoi_gdf.crs)

    print("Searching Sentinel-2 items...")
    try:
        items = search_sentinel2(aoi_union, start, end)
    except Exception as e:
        print(f"Error during STAC search: {e}"); sys.exit(1)

    if not items:
        print("No Sentinel-2 scenes found. Try widening the date window."); sys.exit(1)

    modis_dates = fetch_modis_hotspots(aoi_union, start, end) if use_modis else set()

    dates: List[dt.datetime] = []
    NBR_mean: List[float] = []
    BAI_mean: List[float] = []
    BAIS2_mean: List[float] = []
    NDVI_mean: List[float] = []
    NDWI_mean: List[float] = []
    NDMI_mean: List[float] = []
    NDRE_mean: List[float] = []

    skip_reasons: List[tuple] = []
    BAI_stats: List[Dict[str, Any]] = []
    BAIS2_stats: List[Dict[str, Any]] = []
    NDVI_stats: List[Dict[str, Any]] = []
    NDWI_stats: List[Dict[str, Any]] = []
    NDMI_stats: List[Dict[str, Any]] = []
    NDRE_stats: List[Dict[str, Any]] = []

    idx = index.upper()
    compute_burn = (idx in {"BAI", "BAIS2", "BOTH", "ALL"})
    compute_bai = compute_burn and (idx in {"BAI", "BOTH", "ALL"})
    compute_bais2 = compute_burn and (idx in {"BAIS2", "BOTH", "ALL"})
    compute_agri = (idx in {"AGRI", "ALL"})
    force_nbr = compute_burn

    for it in tqdm(items, desc="Processing Scenes"):
        date = it["properties"]["datetime"][:10]
        try:
            ds = load_clip_item(it, aoi_union)
            clear = build_clear_mask(ds)

            vals = ds["B04"].where(clear).values
            total = vals.size
            good = int(np.isfinite(vals).sum())
            clear_frac = (good / total) if total > 0 else 0.0

            if clear_frac < float(min_clear):
                skip_reasons.append((date, f"clear_frac={round(clear_frac,4)} < min_clear={min_clear}"))
                continue

            clear_ds = ds.where(clear)
            red = clear_ds["B04"]
            nir = clear_ds["B08"]
            swir1 = clear_ds["B11"]
            swir2 = clear_ds["B12"]
            green = clear_ds["B03"]
            red_edge = clear_ds["B05"]

            if force_nbr:
                nbr_da = compute_NBR(nir, swir2)
                nbr = nbr_da.values
                NBR_mean.append(float(np.nanmean(nbr)))

            if compute_bai:
                bai_da = compute_BAI(clear_ds)
                bai = bai_da.values
                BAI_mean.append(float(np.nanmean(bai)))
                finite = np.isfinite(bai); n_bai = int(finite.sum())
                BAI_stats.append({
                    "date": date, "count_valid": n_bai, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(bai)), "std": float(np.nanstd(bai)),
                    "min": float(np.nanmin(bai)) if n_bai > 0 else float("nan"),
                    "p05": _nanpercentile(bai,5), "p25": _nanpercentile(bai,25),
                    "p50": _nanpercentile(bai,50), "p75": _nanpercentile(bai,75),
                    "p95": _nanpercentile(bai,95),
                    "max": float(np.nanmax(bai)) if n_bai > 0 else float("nan"),
                })
                save_png_gray(bai, os.path.join(pics_dir, f"{date}_BAI.png"),
                              viz_scale=viz_scale, viz_min_size=viz_min_size)

                if write_geotiff:
                    out_bai = os.path.join(pics_dir, f"{date}_BAI.tif")
                    bai_out = np.nan_to_num(bai, nan=0.0).astype("float32")
                    with rio.open(
                        out_bai, "w", driver="GTiff",
                        width=bai_da.rio.width, height=bai_da.rio.height,
                        count=1, dtype="float32",
                        crs=bai_da.rio.crs, transform=bai_da.rio.transform()
                    ) as dst:
                        dst.write(bai_out, 1)

            if compute_bais2:
                bais2_da = compute_BAIS2(clear_ds)
                bais2 = bais2_da.values
                BAIS2_mean.append(float(np.nanmean(bais2)))
                finite = np.isfinite(bais2); n_bais2 = int(finite.sum())
                BAIS2_stats.append({
                    "date": date, "count_valid": n_bais2, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(bais2)), "std": float(np.nanstd(bais2)),
                    "min": float(np.nanmin(bais2)) if n_bais2 > 0 else float("nan"),
                    "p05": _nanpercentile(bais2,5), "p25": _nanpercentile(bais2,25),
                    "p50": _nanpercentile(bais2,50), "p75": _nanpercentile(bais2,75),
                    "p95": _nanpercentile(bais2,95),
                    "max": float(np.nanmax(bais2)) if n_bais2 > 0 else float("nan"),
                })

                # ✅ BAIS2 PNG now matches your original blue→yellow→red style
                save_png_bais2_color(
                    bais2,
                    os.path.join(pics_dir, f"{date}_BAIS2.png"),
                    viz_scale=viz_scale,
                    viz_min_size=viz_min_size
                )

                if write_geotiff:
                    out_bais2 = os.path.join(pics_dir, f"{date}_BAIS2.tif")
                    bais2_out = np.nan_to_num(bais2, nan=0.0).astype("float32")
                    with rio.open(
                        out_bais2, "w", driver="GTiff",
                        width=bais2_da.rio.width, height=bais2_da.rio.height,
                        count=1, dtype="float32",
                        crs=bais2_da.rio.crs, transform=bais2_da.rio.transform()
                    ) as dst:
                        dst.write(bais2_out, 1)

            if compute_agri:
                # NDVI
                ndvi_da = compute_NDVI(nir, red)
                ndvi = ndvi_da.values
                NDVI_mean.append(float(np.nanmean(ndvi)))
                finite = np.isfinite(ndvi); n_ndvi = int(finite.sum())
                NDVI_stats.append({
                    "date": date, "count_valid": n_ndvi, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(ndvi)), "std": float(np.nanstd(ndvi)),
                    "min": float(np.nanmin(ndvi)) if n_ndvi > 0 else float("nan"),
                    "p05": _nanpercentile(ndvi,5), "p25": _nanpercentile(ndvi,25),
                    "p50": _nanpercentile(ndvi,50), "p75": _nanpercentile(ndvi,75),
                    "p95": _nanpercentile(ndvi,95),
                    "max": float(np.nanmax(ndvi)) if n_ndvi > 0 else float("nan"),
                })
                save_png_agri_color(ndvi, os.path.join(pics_dir, f"{date}_NDVI.png"),
                                    vmin=-0.2, vmax=0.8, viz_scale=viz_scale, viz_min_size=viz_min_size)

                if write_geotiff:
                    out_ndvi = os.path.join(pics_dir, f"{date}_NDVI.tif")
                    ndvi_out = np.nan_to_num(ndvi, nan=0.0).astype("float32")
                    with rio.open(
                        out_ndvi, "w", driver="GTiff",
                        width=ndvi_da.rio.width, height=ndvi_da.rio.height,
                        count=1, dtype="float32",
                        crs=ndvi_da.rio.crs, transform=ndvi_da.rio.transform()
                    ) as dst:
                        dst.write(ndvi_out, 1)

                # NDWI
                ndwi_da = compute_NDWI(green, nir)
                ndwi = ndwi_da.values
                NDWI_mean.append(float(np.nanmean(ndwi)))
                finite = np.isfinite(ndwi); n_ndwi = int(finite.sum())
                NDWI_stats.append({
                    "date": date, "count_valid": n_ndwi, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(ndwi)), "std": float(np.nanstd(ndwi)),
                    "min": float(np.nanmin(ndwi)) if n_ndwi > 0 else float("nan"),
                    "p05": _nanpercentile(ndwi,5), "p25": _nanpercentile(ndwi,25),
                    "p50": _nanpercentile(ndwi,50), "p75": _nanpercentile(ndwi,75),
                    "p95": _nanpercentile(ndwi,95),
                    "max": float(np.nanmax(ndwi)) if n_ndwi > 0 else float("nan"),
                })
                save_png_agri_color(ndwi, os.path.join(pics_dir, f"{date}_NDWI.png"),
                                    vmin=-0.5, vmax=0.5, viz_scale=viz_scale, viz_min_size=viz_min_size)

                if write_geotiff:
                    out_ndwi = os.path.join(pics_dir, f"{date}_NDWI.tif")
                    ndwi_out = np.nan_to_num(ndwi, nan=0.0).astype("float32")
                    with rio.open(
                        out_ndwi, "w", driver="GTiff",
                        width=ndwi_da.rio.width, height=ndwi_da.rio.height,
                        count=1, dtype="float32",
                        crs=ndwi_da.rio.crs, transform=ndwi_da.rio.transform()
                    ) as dst:
                        dst.write(ndwi_out, 1)

                # NDMI
                ndmi_da = compute_NDMI(nir, swir1)
                ndmi = ndmi_da.values
                NDMI_mean.append(float(np.nanmean(ndmi)))
                finite = np.isfinite(ndmi); n_ndmi = int(finite.sum())
                NDMI_stats.append({
                    "date": date, "count_valid": n_ndmi, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(ndmi)), "std": float(np.nanstd(ndmi)),
                    "min": float(np.nanmin(ndmi)) if n_ndmi > 0 else float("nan"),
                    "p05": _nanpercentile(ndmi,5), "p25": _nanpercentile(ndmi,25),
                    "p50": _nanpercentile(ndmi,50), "p75": _nanpercentile(ndmi,75),
                    "p95": _nanpercentile(ndmi,95),
                    "max": float(np.nanmax(ndmi)) if n_ndmi > 0 else float("nan"),
                })
                save_png_agri_color(ndmi, os.path.join(pics_dir, f"{date}_NDMI.png"),
                                    vmin=-0.5, vmax=0.5, viz_scale=viz_scale, viz_min_size=viz_min_size)

                if write_geotiff:
                    out_ndmi = os.path.join(pics_dir, f"{date}_NDMI.tif")
                    ndmi_out = np.nan_to_num(ndmi, nan=0.0).astype("float32")
                    with rio.open(
                        out_ndmi, "w", driver="GTiff",
                        width=ndmi_da.rio.width, height=ndmi_da.rio.height,
                        count=1, dtype="float32",
                        crs=ndmi_da.rio.crs, transform=ndmi_da.rio.transform()
                    ) as dst:
                        dst.write(ndmi_out, 1)

                # NDRE
                ndre_da = compute_NDRE(nir, red_edge)
                ndre = ndre_da.values
                NDRE_mean.append(float(np.nanmean(ndre)))
                finite = np.isfinite(ndre); n_ndre = int(finite.sum())
                NDRE_stats.append({
                    "date": date, "count_valid": n_ndre, "clear_frac": float(clear_frac),
                    "mean": float(np.nanmean(ndre)), "std": float(np.nanstd(ndre)),
                    "min": float(np.nanmin(ndre)) if n_ndre > 0 else float("nan"),
                    "p05": _nanpercentile(ndre,5), "p25": _nanpercentile(ndre,25),
                    "p50": _nanpercentile(ndre,50), "p75": _nanpercentile(ndre,75),
                    "p95": _nanpercentile(ndre,95),
                    "max": float(np.nanmax(ndre)) if n_ndre > 0 else float("nan"),
                })
                save_png_agri_color(ndre, os.path.join(pics_dir, f"{date}_NDRE.png"),
                                    vmin=-0.2, vmax=0.7, viz_scale=viz_scale, viz_min_size=viz_min_size)

                if write_geotiff:
                    out_ndre = os.path.join(pics_dir, f"{date}_NDRE.tif")
                    ndre_out = np.nan_to_num(ndre, nan=0.0).astype("float32")
                    with rio.open(
                        out_ndre, "w", driver="GTiff",
                        width=ndre_da.rio.width, height=ndre_da.rio.height,
                        count=1, dtype="float32",
                        crs=ndre_da.rio.crs, transform=ndre_da.rio.transform()
                    ) as dst:
                        dst.write(ndre_out, 1)

            save_png_rgb(render_truecolor(clear_ds), os.path.join(pics_dir, f"{date}_truecolor.png"),
                         viz_scale=viz_scale, viz_min_size=viz_min_size)

            if write_geotiff:
                tc = clear_ds
                r = (np.nan_to_num(tc["B04"].values, nan=0.0, posinf=1.0, neginf=0.0) * 10000.0).clip(0,10000).astype("uint16")
                g = (np.nan_to_num(tc["B03"].values, nan=0.0, posinf=1.0, neginf=0.0) * 10000.0).clip(0,10000).astype("uint16")
                b = (np.nan_to_num(tc["B02"].values, nan=0.0, posinf=1.0, neginf=0.0) * 10000.0).clip(0,10000).astype("uint16")

                out_tc = os.path.join(pics_dir, f"{date}_truecolor.tif")
                with rio.open(
                    out_tc, "w", driver="GTiff",
                    width=tc["B04"].rio.width, height=tc["B04"].rio.height,
                    count=3, dtype="uint16",
                    crs=tc["B04"].rio.crs, transform=tc["B04"].rio.transform()
                ) as dst:
                    dst.write(np.stack([r, g, b], axis=0))

            dates.append(dt.datetime.fromisoformat(date))

        except Exception as e:
            skip_reasons.append((date, f"exception: {e}"))
            continue

    if not dates:
        print("Skipped scenes (last 20):")
        for d, why in skip_reasons[-20:]:
            print("  ", d, "->", why)
        print("No valid observations. Try widening dates, lowering --min-clear, or buffering AOI.")
        sys.exit(2)

    order = np.argsort(np.array([d.timestamp() for d in dates]))
    dates = [dates[i] for i in order]
    if force_nbr: NBR_mean = [NBR_mean[i] for i in order]
    if compute_bai: BAI_mean = [BAI_mean[i] for i in order]
    if compute_bais2: BAIS2_mean = [BAIS2_mean[i] for i in order]
    if compute_agri:
        NDVI_mean = [NDVI_mean[i] for i in order]
        NDWI_mean = [NDWI_mean[i] for i in order]
        NDMI_mean = [NDMI_mean[i] for i in order]
        NDRE_mean = [NDRE_mean[i] for i in order]

    BAI_stats   = sorted(BAI_stats, key=lambda r: r["date"])
    BAIS2_stats = sorted(BAIS2_stats, key=lambda r: r["date"])
    NDVI_stats  = sorted(NDVI_stats, key=lambda r: r["date"])
    NDWI_stats  = sorted(NDWI_stats, key=lambda r: r["date"])
    NDMI_stats  = sorted(NDMI_stats, key=lambda r: r["date"])
    NDRE_stats  = sorted(NDRE_stats, key=lambda r: r["date"])

    burn_candidates: List[str] = []
    chosen = "N/A (AGRI only)"
    if compute_burn and (compute_bai or compute_bais2):
        chosen = "BAIS2" if (idx in {"BOTH","ALL"} and compute_bais2) else idx
        if chosen == "BAIS2" and compute_bais2:
            signal = BAIS2_mean
        elif chosen == "BAI" and compute_bai:
            signal = BAI_mean
        else:
            signal = []
            chosen = "N/A (Detection index not computed)"

        if signal:
            burn_candidates = detect_burn_dates_generic(
                signal_mean=signal,
                nbr_mean=NBR_mean,
                dates=dates,
                nbr_drop_gate=-0.03
            )

    reports: Dict[str, Dict[str, str]] = {}
    if compute_bai:
        reports["BAI"] = write_index_report(
            outdir, "BAI",
            "BAI = $\\dfrac{1}{(0.1-\\text{RED})^2 + (0.06-\\text{NIR})^2}$ (RED=B04, NIR=B08)",
            BAI_stats, burn_candidates if chosen == "BAI" else None
        )
    if compute_bais2:
        reports["BAIS2"] = write_index_report(
            outdir, "BAIS2",
            r"$\mathrm{BAIS2} = \left(1-\sqrt{\frac{B06\cdot B07\cdot B8A}{B04}}\right)\left(\frac{B12-B8A}{\sqrt{B12+B8A}}+1\right)$",
            BAIS2_stats, burn_candidates if chosen == "BAIS2" else None
        )
    if compute_agri:
        reports["NDVI"] = write_index_report(
            outdir, "NDVI",
            r"$\mathrm{NDVI} = \dfrac{B08-B04}{B08+B04}$ (NIR=B08, RED=B04)",
            NDVI_stats, notes_suffix="Measures vegetation greenness and biomass."
        )
        reports["NDWI"] = write_index_report(
            outdir, "NDWI",
            r"$\mathrm{NDWI} = \dfrac{B03-B08}{B03+B08}$ (GREEN=B03, NIR=B08)",
            NDWI_stats, notes_suffix="McFeeters version. Highlights open water bodies."
        )
        reports["NDMI"] = write_index_report(
            outdir, "NDMI",
            r"$\mathrm{NDMI} = \dfrac{B08-B11}{B08+B11}$ (NIR=B08, SWIR1=B11)",
            NDMI_stats, notes_suffix="Measures vegetation water content (moisture/stress)."
        )
        reports["NDRE"] = write_index_report(
            outdir, "NDRE",
            r"$\mathrm{NDRE} = \dfrac{B08-B05}{B08+B05}$ (NIR=B08, RED\_EDGE=B05)",
            NDRE_stats, notes_suffix="Sensitive to chlorophyll content in mid/late-season crops."
        )

    pics = sorted([f for f in os.listdir(pics_dir) if f.endswith((".png",".tif"))])
    out = {
        "inputs": {
            "start": start, "end": end, "ask": list(ask),
            "aoi": str(aoi), "min_clear": float(min_clear),
            "index": index.upper(), "write_geotiff": bool(write_geotiff),
            "viz-scale": int(viz_scale), "viz-min-size": int(viz_min_size),
        },
        "burnDateList": burn_candidates,
        "burnDatePics": pics,
        "timeseries": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            **({"NBR_mean": NBR_mean} if force_nbr else {}),
            **({"BAI_mean": BAI_mean} if compute_bai else {}),
            **({"BAIS2_mean": BAIS2_mean} if compute_bais2 else {}),
            **({"NDVI_mean": NDVI_mean} if compute_agri else {}),
            **({"NDWI_mean": NDWI_mean} if compute_agri else {}),
            **({"NDMI_mean": NDMI_mean} if compute_agri else {}),
            **({"NDRE_mean": NDRE_mean} if compute_agri else {}),
        },
        "reports": {k: {"csv": v["csv"], "markdown": v["markdown"]} for k, v in reports.items()},
        "notes": {
            "detection_index_used": chosen,
            "indices_computed": idx,
            "masking": "SCL clear classes 4,5,6,7; resampled to 10 m",
            "modis_hotspots_found": [d.strftime("%Y-%m-%d") for d in modis_dates]
        }
    }
    with open(os.path.join(outdir, "burn_dates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n✅ Processing complete.")
    print(f"  JSON Metadata: {os.path.join(outdir, 'burn_dates.json')}")
    print(f"  Images and Rasters: {pics_dir}")
    for k, v in reports.items():
        print(f"  {k} CSV: {v['csv']}")
        print(f"  {k} MD: {v['markdown']}")
    if modis_dates:
        print(f"  🔥 MODIS Hotspots found: {sorted([d.strftime('%Y-%m-%d') for d in modis_dates])}")


# ---------------------------
# Post-run Event Summary (MD + PDF) — post-processing only
# ---------------------------

def _parse_outdir_from_argv(argv: List[str]) -> Optional[str]:
    """Extract --out <dir> (or --out=<dir>) from argv."""
    if not argv:
        return None
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--out="):
            return a.split("=", 1)[1]
    return None

def _first_diff(series: List[float]) -> List[float]:
    if not series or len(series) < 2:
        return []
    arr = np.array(series, dtype=float)
    return list(arr[1:] - arr[:-1])

def _safe_get(ts: Dict[str, Any], key: str) -> List[float]:
    v = ts.get(key)
    return v if isinstance(v, list) else []

def _pick_event_anchor(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose an event anchor date for the summary.
    Priority:
      1) first burnDateList if exists
      2) max positive jump in BAIS2_mean (if present)
      3) max positive jump in BAI_mean (if present)
    """
    ts = meta.get("timeseries", {}) or {}
    dates = ts.get("dates", []) or []
    burn_list = meta.get("burnDateList", []) or []

    if burn_list:
        return {"date": burn_list[0], "how": "burnDateList_first"}

    bais2 = _safe_get(ts, "BAIS2_mean")
    if bais2 and len(bais2) == len(dates):
        steps = np.array(_first_diff(bais2), dtype=float)
        if steps.size:
            i = int(np.nanargmax(steps)) + 1
            return {"date": dates[i], "how": "BAIS2_max_jump", "index": i}

    bai = _safe_get(ts, "BAI_mean")
    if bai and len(bai) == len(dates):
        steps = np.array(_first_diff(bai), dtype=float)
        if steps.size:
            i = int(np.nanargmax(steps)) + 1
            return {"date": dates[i], "how": "BAI_max_jump", "index": i}

    return {"date": dates[-1] if dates else None, "how": "fallback_last"}

def _event_steps(meta: Dict[str, Any], event_date: Optional[str]) -> Dict[str, Optional[float]]:
    ts = meta.get("timeseries", {}) or {}
    dates = ts.get("dates", []) or []
    if not event_date or event_date not in dates:
        return {}

    i = dates.index(event_date)
    if i <= 0:
        return {}

    def step(name: str) -> Optional[float]:
        s = _safe_get(ts, name)
        if not s or len(s) != len(dates):
            return None
        try:
            return float(s[i] - s[i-1])
        except Exception:
            return None

    return {
        "BAIS2_step": step("BAIS2_mean"),
        "BAI_step": step("BAI_mean"),
        "NBR_step": step("NBR_mean"),
        "NDVI_step": step("NDVI_mean"),
        "NDMI_step": step("NDMI_mean"),
        "NDWI_step": step("NDWI_mean"),
    }

def _burned_or_not(steps: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """
    Simple scoring for summary (post-processing only).
    Uses step signatures:
      - BAIS2 up, BAI up, NBR down, NDVI down, NDMI down
    Produces burned True/False and confidence 0-100.
    """
    score = 0.0
    reasons: List[str] = []

    bais2 = steps.get("BAIS2_step")
    bai = steps.get("BAI_step")
    nbr = steps.get("NBR_step")
    ndvi = steps.get("NDVI_step")
    ndmi = steps.get("NDMI_step")
    ndwi = steps.get("NDWI_step")

    if bais2 is not None and np.isfinite(bais2) and bais2 > 0.03:
        score += 1.5; reasons.append(f"BAIS2 increased ({bais2:+.4f}) at event date.")
    if bai is not None and np.isfinite(bai) and bai > 0.5:
        score += 1.5; reasons.append(f"BAI spiked ({bai:+.4f}) at event date.")
    if nbr is not None and np.isfinite(nbr) and nbr < -0.05:
        score += 2.0; reasons.append(f"NBR dropped ({nbr:+.4f}) consistent with burn scar.")
    if ndvi is not None and np.isfinite(ndvi) and ndvi < -0.08:
        score += 1.5; reasons.append(f"NDVI dropped ({ndvi:+.4f}) consistent with vegetation loss.")
    if ndmi is not None and np.isfinite(ndmi) and ndmi < -0.05:
        score += 1.0; reasons.append(f"NDMI dropped ({ndmi:+.4f}) consistent with moisture loss.")
    if ndwi is not None and np.isfinite(ndwi) and abs(ndwi) > 0.05:
        score += 0.5; reasons.append(f"NDWI changed ({ndwi:+.4f}) suggesting water/background change (context).")

    burned = score >= 4.0
    conf = int(np.clip((score / 6.0) * 100.0, 0, 100))

    if burned:
        reasons.insert(0, "Overall signal combination is consistent with a burn event.")
    else:
        reasons.insert(0, "Overall signal combination is NOT strongly consistent with a burn event.")

    return {"burned": bool(burned), "confidence": int(conf), "reasons": reasons}

def _infer_tillage_date(meta: Dict[str, Any], burn_date: Optional[str]) -> Dict[str, Any]:
    """
    Optional tillage inference: biggest NDVI drop not near burn and without burn spikes.
    Post-processing only.
    """
    ts = meta.get("timeseries", {}) or {}
    dates = ts.get("dates", []) or []
    ndvi = _safe_get(ts, "NDVI_mean")
    bai = _safe_get(ts, "BAI_mean")
    bais2 = _safe_get(ts, "BAIS2_mean")

    if not dates or not ndvi or len(ndvi) != len(dates) or len(ndvi) < 2:
        return {"date": None, "how": "unavailable"}

    burn_idx = dates.index(burn_date) if (burn_date and burn_date in dates) else None
    steps = np.array(_first_diff(ndvi), dtype=float)  # len n-1
    order = np.argsort(steps)  # most negative first

    for j in order:
        i = int(j) + 1
        if burn_idx is not None and abs(i - burn_idx) <= 3:
            continue
        if steps[j] > -0.12:
            break

        # gate out burn-like spikes
        ok = True
        if bai and len(bai) == len(dates):
            if np.isfinite(bai[i-1]) and bai[i-1] != 0 and np.isfinite(bai[i]) and (bai[i] / bai[i-1]) > 1.20:
                ok = False
        if bais2 and len(bais2) == len(dates):
            if np.isfinite(bais2[i-1]) and np.isfinite(bais2[i]) and (bais2[i] - bais2[i-1]) > 0.03:
                ok = False
        if not ok:
            continue

        return {"date": dates[i], "how": "ndvi_step_drop_no_burn_spike", "ndvi_step": float(steps[j])}

    return {"date": None, "how": "no_confident_tillage_found"}

def _calc_aoi_area_ha(meta: Dict[str, Any]) -> Optional[float]:
    try:
        inputs = meta.get("inputs", {}) or {}
        aoi_raw = inputs.get("aoi")
        if not aoi_raw:
            return None
        gdf = load_aoi(str(aoi_raw))
        union = gpd.GeoDataFrame(geometry=[unary_union(gdf.geometry)], crs=gdf.crs)
        area_ha = float(union.to_crs(3857).area.sum() / 10000.0)
        return round(area_ha, 2)
    except Exception:
        return None

def write_event_summary_md(outdir: str, summary: Dict[str, Any]) -> str:
    path = os.path.join(outdir, "event_summary.md")
    ctx = summary.get("context", {})
    anchor = summary.get("anchor", {})
    burn = summary.get("burn", {})
    steps = summary.get("steps", {})
    till = summary.get("tillage", {})

    lines: List[str] = []
    lines.append("Event Summary Report\n")
    lines.append("Context\n\n")
    for k in ["start", "end", "index_mode", "detection_index_used", "scenes_processed", "aoi_area_ha"]:
        if k in ctx:
            lines.append(f"{k.replace('_',' ').title()}: {ctx.get(k)}\n")
    lines.append("\nEvent anchor\n\n")
    lines.append(f"Event date used: {anchor.get('date')}\n")
    lines.append(f"How chosen: {anchor.get('how')}\n")

    lines.append("\nBurned or not?\n\n")
    lines.append(f"Burned: {burn.get('burned')}\n")
    lines.append(f"Confidence (0-100): {burn.get('confidence')}\n\n")
    lines.append("Reasons:\n\n")
    for r in burn.get("reasons", []) or []:
        lines.append(f"- {r}\n")

    lines.append("\nKey index step changes at event date\n\n")
    for k in ["BAIS2_step","BAI_step","NBR_step","NDVI_step","NDMI_step","NDWI_step"]:
        if k in steps and steps[k] is not None and np.isfinite(steps[k]):
            lines.append(f"{k.replace('_',' ')}: {steps[k]:+.4f}\n")
        else:
            lines.append(f"{k.replace('_',' ')}: None\n")

    lines.append("\nTillage / disturbance (optional)\n\n")
    lines.append(f"Tillage date: {till.get('date')}\n")
    lines.append(f"How chosen: {till.get('how')}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return path

def write_event_summary_pdf(outdir: str, summary: Dict[str, Any]) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.units import cm

    path = os.path.join(outdir, "event_summary.pdf")
    c = pdf_canvas.Canvas(path, pagesize=A4)
    w, h = A4

    x = 2 * cm
    y = h - 2 * cm
    line_h = 14

    def draw(txt, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 12 if bold else 11)
        c.drawString(x, y, txt)
        y -= line_h
        if y < 2 * cm:
            c.showPage()
            y = h - 2 * cm

    ctx = summary.get("context", {})
    anchor = summary.get("anchor", {})
    burn = summary.get("burn", {})
    steps = summary.get("steps", {})
    till = summary.get("tillage", {})

    draw("Event Summary Report", bold=True)
    y -= 6

    draw("Context", bold=True)
    for k in ["start", "end", "index_mode", "detection_index_used", "scenes_processed", "aoi_area_ha"]:
        if k in ctx:
            draw(f"{k.replace('_',' ').title()}: {ctx.get(k)}")
    y -= 6

    draw("Event anchor", bold=True)
    draw(f"Event date used: {anchor.get('date')}")
    draw(f"How chosen: {anchor.get('how')}")
    y -= 6

    draw("Burned or not?", bold=True)
    draw(f"Burned: {burn.get('burned')}")
    draw(f"Confidence (0-100): {burn.get('confidence')}")
    draw("Reasons:", bold=True)
    for r in burn.get("reasons", []) or []:
        draw(f"- {r}")
    y -= 6

    draw("Key index step changes at event date", bold=True)
    for k in ["BAIS2_step","BAI_step","NBR_step","NDVI_step","NDMI_step","NDWI_step"]:
        v = steps.get(k)
        if v is not None and np.isfinite(v):
            draw(f"{k.replace('_',' ')}: {v:+.4f}")
        else:
            draw(f"{k.replace('_',' ')}: None")
    y -= 6

    draw("Tillage / disturbance (optional)", bold=True)
    draw(f"Tillage date: {till.get('date')}")
    draw(f"How chosen: {till.get('how')}")

    c.save()
    return path

def generate_event_summary_files(outdir: str) -> Dict[str, Any]:
    meta_path = os.path.join(outdir, "burn_dates.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"burn_dates.json not found in: {outdir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    anchor = _pick_event_anchor(meta)
    steps = _event_steps(meta, anchor.get("date"))
    burn = _burned_or_not(steps)
    till = _infer_tillage_date(meta, anchor.get("date") if burn.get("burned") else None)

    inputs = meta.get("inputs", {}) or {}
    notes = meta.get("notes", {}) or {}
    ts = meta.get("timeseries", {}) or {}

    ctx = {
        "start": inputs.get("start"),
        "end": inputs.get("end"),
        "index_mode": notes.get("indices_computed"),
        "detection_index_used": notes.get("detection_index_used"),
        "scenes_processed": len(ts.get("dates") or []),
    }
    area_ha = _calc_aoi_area_ha(meta)
    if area_ha is not None:
        ctx["aoi_area_ha"] = area_ha

    summary = {"context": ctx, "anchor": anchor, "burn": burn, "steps": steps, "tillage": till}
    md_path = write_event_summary_md(outdir, summary)
    pdf_path = write_event_summary_pdf(outdir, summary)

    try:
        meta["event_summary"] = summary
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    return {"summary": summary, "md": md_path, "pdf": pdf_path}


if __name__ == "__main__":
    exit_code = 0
    try:
        main(standalone_mode=False)
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else 0
    finally:
        outdir = _parse_outdir_from_argv(sys.argv)
        if outdir:
            try:
                res = generate_event_summary_files(outdir)
                print(f"  📄 Event Summary MD:  {res['md']}")
                print(f"  📄 Event Summary PDF: {res['pdf']}")
            except Exception as _e:
                print(f"  ⚠️ Could not generate event summary report: {_e}")

    if exit_code:
        sys.exit(exit_code)
