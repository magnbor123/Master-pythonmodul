#!/usr/bin/env python3
"""
coastal_vertical_profiles.py

Generate combined vertical profiles (temperature and salinity vs. depth) for ARGO casts
that lie near the Norwegian and Svalbard coast, comparing ARGO measurements to multiple
SINMOD model variants (standard, nudged, assimilated). For each month and each qualifying
cast:

  1. Identify ARGO casts with depth ≥ VERT_DEPTH_MIN and within VERT_COAST_KM of the coast.
  2. Ensure the cast appears in all relevant SINMOD‐comparison CSVs (one per variant).
  3. Merge ARGO and SINMOD model values for that cast and depth range.
  4. Determine the shallowest maximum depth common to all available model variants.
  5. Load the nearest Copernicus SST for the cast date/location.
  6. Plot ARGO, standard, nudged, and assimilated temperature and salinity profiles
     on a dual‐axis figure, marking the SST at depth = 0.
  7. Save each plot to a structured directory under PROFILE_BASE.

Configuration is imported from config.py:
  - OUTPUT_DIRS:      dict mapping each SINMOD variant → its CSV directory
  - YEARS, MONTHS:    lists of years and months to process
  - VERT_DEPTH_MIN:   minimum ARGO depth (m) to include in “vertical” plots
  - VERT_COAST_KM:    maximum distance (km) from coast for “coastal” casts
  - VERT_PROFILE_BASE: base directory for saving profile plots
  - SST_DIR:          directory containing Copernicus SST NetCDF files (used to fetch an SST value)

Dependencies:
    pandas, numpy, matplotlib, xarray, shapely, cartopy, config.py
"""

import glob
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config


# ─── UNPACK CONFIG ─────────────────────────────────────────────────────────────
OUTPUT_DIRS      = config.OUTPUT_DIRS      # {variant: path/to/csvs}
YEARS            = config.YEARS            # [2019, 2022, 2023]
MONTHS           = config.MONTHS           # [1, 2, ..., 12]
VERT_DEPTH_MIN   = config.VERT_DEPTH_MIN   # e.g. 50.0 meters
VERT_COAST_KM    = config.VERT_COAST_KM    # e.g. 10.0 km
PROFILE_BASE     = config.VERT_PROFILE_BASE  # base directory for profile plots
SST_DIR          = config.SST_DIR           # Copernicus SST directory

# ─── BUILD CLIPPED NORWAY+SVALBARD COASTLINE ──────────────────────────────────
# We load the Natural Earth coastline, clip to the bounding box of Norway+Svalbard,
# and prepare a Shapely geometry for fast "distance to coast" checks.
coast_feat  = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
coast_union = unary_union(list(coast_feat.geometries()))
clip_box    = box(-10.0, 58.0, 40.0, 82.0)  # lon_min, lat_min, lon_max, lat_max
nor_coast   = coast_union.intersection(clip_box)
prep_coast  = prep(nor_coast)


def dist_to_coast_km(lon: float, lat: float) -> float:
    """
    Approximate distance (in kilometers) from a geographic point (lon, lat)
    to the clipped Norway+Svalbard coastline.

    Parameters
    ----------
    lon : float
        Longitude of the point (in degrees).
    lat : float
        Latitude of the point (in degrees).

    Returns
    -------
    float
        Distance to the nearest point on nor_coast, in kilometers (approx. using 1° ≈ 111 km).
    """
    # Shapely distance returns degrees; multiply by ~111 to convert to kilometers.
    return nor_coast.distance(Point(lon, lat)) * 111.0


# ─── SST LOADER ────────────────────────────────────────────────────────────────
def load_sst_value(date: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Load the nearest Copernicus SST value (°C) at a given date and location (lat, lon).

    1. Determine which Copernicus file to open based on year and month:
       - If month ≤ 8, use "*jan*aug*.nc"; else use "*sep*.nc" for that year.
    2. Load the SST dataset (converted from Kelvin to °C).
    3. Find the time index closest to 'date'.
    4. Interpolate spatially by nearest neighbor (index of closest lat & lon).
    5. Return the SST value at that index.

    Parameters
    ----------
    date : pd.Timestamp
        The date/time for which to retrieve an SST.
    lat : float
        Latitude of the ARGO cast.
    lon : float
        Longitude of the ARGO cast.

    Returns
    -------
    float or None
        SST in °C at the nearest time & location, or None if no file is found.
    """
    year, month = date.year, date.month

    # Select file pattern based on month
    patt = "*jan*aug*.nc" if month <= 8 else "*sep*.nc"
    files = glob.glob(str(SST_DIR / f"*{year}*{patt}"))
    if not files:
        return None

    # Open the first matching file (there should be exactly one)
    ds = xr.open_dataset(files[0], engine="h5netcdf")
    # Convert Kelvin → °C
    sst_all = ds["analysed_sst"].values - 273.15  # shape: (time, lat, lon) or (lat, lon)
    times   = pd.to_datetime(ds["time"].values)

    # Find the index of the SST timestamp closest to 'date'
    idx_time = np.abs(times - date).argmin()
    # If sst_all is 3D (time, lat, lon), take that slice; otherwise assume 2D year‐round
    slice_ = sst_all[idx_time] if sst_all.ndim == 3 else sst_all

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()

    # Find nearest lat/lon indices
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()

    return float(slice_[i, j])


# ─── KEY LOADER ────────────────────────────────────────────────────────────────
def load_keys(path: Path) -> pd.DataFrame:
    """
    Load the minimal set of columns ("Day", "ARGO Lat", "ARGO Lon", "ARGO Depth")
    from a monthly ARGO–SINMOD CSV, coerce types, round values, and group duplicates.

    1. Read 'Day' as datetime, coerce invalid strings to NaT.
    2. Round 'ARGO Lat' & 'ARGO Lon' to 0.001°, 'ARGO Depth' to 0.1 m.
    3. Group by (Day, ARGO Lat, ARGO Lon) and take the maximum ARGO Depth for each group,
       returning a DataFrame keyed by unique casts (date, lat, lon).

    Parameters
    ----------
    path : Path
        Path to the CSV file (e.g. "standard_201901.csv").

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["Day", "ARGO Lat", "ARGO Lon", "ARGO Depth"] and
        one row per unique (Day, Lat, Lon), taking the deepest cast if duplicates exist.
    """
    df = pd.read_csv(path, usecols=["Day", "ARGO Lat", "ARGO Lon", "ARGO Depth"])

    # Convert 'Day' column to pandas datetime
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

    # Round lat/lon/depth to reduce floating‐point noise
    df["ARGO Lat"]   = pd.to_numeric(df["ARGO Lat"], errors="coerce").round(3)
    df["ARGO Lon"]   = pd.to_numeric(df["ARGO Lon"], errors="coerce").round(3)
    df["ARGO Depth"] = pd.to_numeric(df["ARGO Depth"], errors="coerce").round(1)

    # Group by (Day, Lat, Lon) and take max depth
    return df.groupby(["Day", "ARGO Lat", "ARGO Lon"], as_index=False)["ARGO Depth"].max()


# ─── FULL PROFILE LOADER ───────────────────────────────────────────────────────
def load_and_rename(path: Path, sim: str) -> pd.DataFrame:
    """
    Load a full ARGO–SINMOD comparison CSV for one variant and rename
    the SINMOD columns to include the variant name.

    1. Read the CSV into a DataFrame.
    2. Convert 'Day' to datetime, and round numeric columns.
    3. Rename "Temp SINMOD (°C)" → "Temp {sim} (°C)" and 
       "Salinity SINMOD (PSU)" → "Salinity {sim} (PSU)".

    Parameters
    ----------
    path : Path
        Path to the CSV (e.g. "standard_201901.csv").
    sim : str
        Variant name, one of "standard", "nudged", "assimilated".

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with renamed SINMOD columns and rounded numeric values.
    """
    df = pd.read_csv(path)
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

    # Determine renaming dictionary based on variant
    if sim == "standard":
        old = {
            "Temp SINMOD (°C)":      "Temp standard (°C)",
            "Salinity SINMOD (PSU)": "Salinity standard (PSU)",
        }
    else:
        old = {
            "Temp SINMOD (°C)":      f"Temp {sim} (°C)",
            "Salinity SINMOD (PSU)": f"Salinity {sim} (PSU)",
        }
    df = df.rename(columns=old)

    # Round numeric columns to reduce floating‐point noise
    for c, p in [
        ("ARGO Lat", 3),
        ("ARGO Lon", 3),
        ("ARGO Depth", 1),
        ("Temp ARGO (°C)", 2),
        ("Salinity ARGO (PSU)", 2)
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(p)

    return df


# ─── PROFILE PLOTTING ──────────────────────────────────────────────────────────
def plot_combined_profile(group: pd.DataFrame, outpath: Path, sst: float = None):
    """
    Plot combined temperature and salinity vertical profiles for ARGO + SINMOD variants.

    1. Sort the group by 'ARGO Depth'.
    2. Extract arrays for:
       - depth
       - ARGO T/S
       - SINMOD T/S for each variant (if present)
    3. Create a 1×2 subplot (temperature on left, salinity on right):
       - Plot ARGO profile in black.
       - Plot "standard" profile in green, "nudged" in blue, "assimilated" in red if available.
       - If SST is provided, scatter a single dot at (SST, depth=0).
       - Invert y‐axis so surface = 0 is at top.
       - Add legends, gridlines, titles, and axis labels.
    4. Save figure to 'outpath'.

    Parameters
    ----------
    group : pd.DataFrame
        A DataFrame corresponding to one ARGO cast (unique Day, Lat, Lon) with columns:
          - 'ARGO Depth', 'Temp ARGO (°C)', 'Salinity ARGO (PSU)'
          - 'Temp standard (°C)', 'Temp nudged (°C)', 'Temp assimilated (°C)' (if present)
          - 'Salinity standard (PSU)', 'Salinity nudged (PSU)', 'Salinity assimilated (PSU)' (if present)
    outpath : Path
        Path (including filename) where the PNG will be saved.
    sst : float or None
        Surface temperature (°C) from Copernicus at depth=0; if provided, a scatter point is plotted.

    Returns
    -------
    None
    """
    # Sort by ARGO Depth ascending
    group = group.sort_values("ARGO Depth")
    depth  = group["ARGO Depth"].values
    t_argo = group["Temp ARGO (°C)"].values
    s_argo = group["Salinity ARGO (PSU)"].values

    # Extract SINMOD temperature/salinity for each variant (fill with NaN if absent)
    t_std = group.get("Temp standard (°C)",      np.full_like(depth, np.nan))
    t_nud = group.get("Temp nudged (°C)",        np.full_like(depth, np.nan))
    t_ass = group.get("Temp assimilated (°C)",   np.full_like(depth, np.nan))
    s_std = group.get("Salinity standard (PSU)", np.full_like(depth, np.nan))
    s_nud = group.get("Salinity nudged (PSU)",   np.full_like(depth, np.nan))
    s_ass = group.get("Salinity assimilated (PSU)", np.full_like(depth, np.nan))

    # Create figure with two subplots: temperature (ax0) and salinity (ax1)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Temperature subplot (ax0) ---
    handles, labels = [], []
    # Plot ARGO in black
    h = ax0.plot(t_argo, depth, 'k-', label="ARGO")[0]
    handles.append(h); labels.append("ARGO")
    # Plot each model variant if available
    if np.any(~np.isnan(t_std)):
        h = ax0.plot(t_std, depth, 'g-', label="Standard")[0]
        handles.append(h); labels.append("Standard")
    if np.any(~np.isnan(t_nud)):
        h = ax0.plot(t_nud, depth, 'b-', label="Nudged")[0]
        handles.append(h); labels.append("Nudged")
    if np.any(~np.isnan(t_ass)):
        h = ax0.plot(t_ass, depth, 'r-', label="Assimilated")[0]
        handles.append(h); labels.append("Assimilated")

    # If SST (surface temperature) is provided, plot as a scatter at depth=0
    if sst is not None:
        h = ax0.scatter(
            [sst], [0], s=50, edgecolor='k',
            facecolor='yellow', label="SST", zorder=10
        )
        handles.append(h); labels.append("SST")

    ax0.invert_yaxis()  # so depth increases downward
    ax0.set_xlim(-2, 20)
    ax0.set_xlabel("Temperature (°C)")
    ax0.set_ylabel("Depth (m)")
    ax0.set_title("Temperature Profile")
    ax0.grid(True, linestyle='--', linewidth=0.5)
    ax0.legend(loc="upper left")

    # --- Salinity subplot (ax1) ---
    handles, labels = [], []
    h = ax1.plot(s_argo, depth, 'k-', label="ARGO")[0]
    handles.append(h); labels.append("ARGO")
    if np.any(~np.isnan(s_std)):
        h = ax1.plot(s_std, depth, 'g-', label="Standard")[0]
        handles.append(h); labels.append("Standard")
    if np.any(~np.isnan(s_nud)):
        h = ax1.plot(s_nud, depth, 'b-', label="Nudged")[0]
        handles.append(h); labels.append("Nudged")
    if np.any(~np.isnan(s_ass)):
        h = ax1.plot(s_ass, depth, 'r-', label="Assimilated")[0]
        handles.append(h); labels.append("Assimilated")

    ax1.invert_yaxis()
    ax1.set_xlabel("Salinity (PSU)")
    ax1.set_title("Salinity Profile")
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.legend(loc="upper left")

    # --- Overall figure title and save ---
    day = group["Day"].dt.strftime("%Y-%m-%d").iloc[0]
    lat, lon = group
