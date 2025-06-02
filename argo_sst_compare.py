#!/usr/bin/env python3
"""
argo_sst_compare.py

Generate “flat” CSV files comparing ARGO surface temperatures to Copernicus SST
for each month. The CSV for a given month contains one row per ARGO surface cast,
with the cast’s date, latitude, longitude, ARGO surface temperature, interpolated
Copernicus SST at the same time and location, and the difference (ARGO – SST).

Configuration (paths and parameters) is imported from config.py:
  - ARGO_DIR: directory where ARGO NetCDF files are stored
  - SST_DIR: directory where Copernicus SST NetCDF files are stored
  - TIME_TOL_HOURS: ± tolerance (in hours) for matching ARGO cast time to SST snapshot
  - OUTPUT_BASE: base path for all outputs
This script writes its monthly comparison CSVs into OUTPUT_BASE/argo_sst_flat/.

Usage:
    python argo_sst_compare.py

Dependencies:
    numpy, pandas, xarray, scipy, pyproj, config.py
"""

import os
import re
import numpy as np
import pandas as pd
import xarray as xr

from datetime import timedelta
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

import config

# ------------------------------------------------------------------
# Paths (from config.py)
# ------------------------------------------------------------------
ARGO_DIR        = config.ARGO_DIR         # Path to ARGO NetCDF files
SST_DIR         = config.SST_DIR          # Path to Copernicus SST NetCDF files
TIME_TOL_HOURS  = config.TIME_TOL_HOURS   # ± tolerance (hours) for time‐matching ARGO to SST

# Directory where the “flat” ARGO‐SST CSVs will be written
ARGO_SST_FLAT_DIR = config.OUTPUT_BASE / "argo_sst_flat"
ARGO_SST_FLAT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helper: Open an ARGO NetCDF file safely (try multiple engines)
# ------------------------------------------------------------------
def open_argo_dataset(path: Path) -> xr.Dataset:
    """
    Attempt to open an ARGO NetCDF dataset with xarray, trying multiple engines in order:
      1) 'netcdf4'
      2) 'scipy'
      3) 'h5netcdf'
    If none succeed, prints an error and returns None.

    Parameters
    ----------
    path : Path
        Full path to the ARGO NetCDF file.

    Returns
    -------
    xr.Dataset or None
        The opened xarray Dataset if successful, otherwise None.
    """
    p = Path(path)
    if not p.exists():
        print(f"  ARGO file not found: {p}. Skipping.")
        return None

    for engine in ("netcdf4", "scipy", "h5netcdf"):
        try:
            return xr.open_dataset(p.as_posix(), engine=engine)
        except Exception:
            pass

    print(f"  Could not open {p.name} with any engine; skipping.")
    return None


# ------------------------------------------------------------------
# Load all Copernicus SST files into memory
# ------------------------------------------------------------------
print("Loading all Copernicus SST from:", SST_DIR)
ds_sst = xr.open_mfdataset(
    str(SST_DIR / "*.nc"),
    combine="by_coords",
    engine="h5netcdf",
    chunks={}   # empty dict ⇒ load everything into memory
)

# Convert SST from Kelvin to °C in place
ds_sst["analysed_sst"] -= 273.15

# Extract time, latitude, and longitude arrays for interpolation
sst_times = pd.to_datetime(ds_sst["time"].values)
sst_lats  = ds_sst["latitude"].values
sst_lons  = ds_sst["longitude"].values


# ------------------------------------------------------------------
# Extract surface ARGO casts for a given (year, month)
# ------------------------------------------------------------------
def process_argo(ds_a: xr.Dataset, year: int, month: int) -> pd.DataFrame:
    """
    Extract the surface temperature and location of all ARGO casts in the specified year/month.

    Parameters
    ----------
    ds_a : xr.Dataset
        An xarray Dataset containing ARGO profile data for one month file. Must have:
          - 'JULD' (cast times)
          - 'LATITUDE', 'LONGITUDE'
          - 'DEPH_CORRECTED' (corrected depths, possibly 2D array)
          - 'TEMP' (temperature, possibly 2D array)
    year : int
        Calendar year of interest (e.g., 2019).
    month : int
        Calendar month of interest (1–12).

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per ARGO cast in that month, containing:
          - 'Day'              : cast date floored to midnight
          - 'Time'             : full cast timestamp
          - 'ARGO Lat'         : latitude (rounded to 0.001°)
          - 'ARGO Lon'         : longitude (rounded to 0.001°)
          - 'Temp ARGO (°C)'   : surface temperature (rounded to 0.01°C)
        If no casts exist in the specified month, returns an empty DataFrame.
    """
    # Convert ARGO time variable 'JULD' into pandas timestamps
    times_all = pd.to_datetime(ds_a["JULD"].values)

    # Build a boolean mask for the specified year & month
    mask = (times_all.year == year) & (times_all.month == month)
    if not mask.any():
        return pd.DataFrame()  # No ARGO casts in this year/month

    # Filter lat, lon, and times using the mask
    lats  = ds_a["LATITUDE"].values[mask]
    lons  = ds_a["LONGITUDE"].values[mask]
    times = times_all[mask]

    # Extract corrected depths and temperatures
    depths_all = ds_a["DEPH_CORRECTED"].values
    temps_all  = ds_a["TEMP"].values if "TEMP" in ds_a else np.full_like(depths_all, np.nan)

    # If each cast has multiple depth levels, pick the shallowest level
    if depths_all.ndim > 1:
        Dm  = depths_all[mask, :]  # shape: (n_casts_in_month, n_levels)
        idx = np.nanargmin(Dm, axis=1)  # index of shallowest depth for each cast
        surf_depth = Dm[np.arange(idx.size), idx]

        Tm = temps_all[mask, :]
        # Extract the temperature at the shallowest level, or NaN if no valid index
        surf_temp = np.array([
            Tm[i, idx[i]] if not np.isnan(idx[i]) else np.nan
            for i in range(idx.size)
        ])
    else:
        # If only a single depth per cast, use it directly
        surf_depth = depths_all[mask]
        surf_temp  = temps_all[mask]

    # Build a DataFrame with the surface‐cast information
    df = pd.DataFrame({
        "Day":            times.floor("D"),               # date floored to midnight
        "Time":           times,                          # full timestamp
        "ARGO Lat":       np.round(lats, 3),              # rounded latitude
        "ARGO Lon":       np.round(lons, 3),              # rounded longitude
        "Temp ARGO (°C)": np.round(surf_temp, 2)          # rounded temperature in °C
    }).dropna(subset=["Temp ARGO (°C)"])                  # drop rows where temperature is NaN

    return df


# ------------------------------------------------------------------
# Vectorized time‐matching: for each ARGO timestamp, find nearest SST timestamp index within ±tol
# ------------------------------------------------------------------
def vectorized_time_match(argo_times: pd.DatetimeIndex,
                          sst_times: pd.DatetimeIndex,
                          tol: timedelta = timedelta(hours=TIME_TOL_HOURS)
                         ) -> np.ndarray:
    """
    For each ARGO cast time, find the index of the nearest SST snapshot time (within ±tol).
    If the nearest SST time is farther than tol, mark that cast with index -1.

    Parameters
    ----------
    argo_times : pd.DatetimeIndex
        Timestamps of ARGO surface casts (length N_argo).
    sst_times : pd.DatetimeIndex
        Timestamps of Copernicus SST snapshots (length N_sst).
    tol : timedelta, optional
        Maximum allowed difference. Default is ±TIME_TOL_HOURS.

    Returns
    -------
    np.ndarray
        Integer array of length N_argo. Each element is the index (0..N_sst-1) of
        the closest SST time, or -1 if no time lies within ±tol.
    """
    # Convert times to integer nanoseconds since epoch for vectorized diff
    a_ns   = argo_times.values.astype("datetime64[ns]").astype(np.int64)  # shape: (N_argo,)
    s_ns   = sst_times.values.astype("datetime64[ns]").astype(np.int64)   # shape: (N_sst,)
    tol_ns = np.timedelta64(tol).astype("timedelta64[ns]").astype(np.int64)

    # Compute absolute time differences between each ARGO and every SST time
    diff   = np.abs(a_ns[:, None] - s_ns[None, :])  # shape: (N_argo, N_sst)

    # For each ARGO index, find SST index with minimal time difference
    idx    = diff.argmin(axis=1)                    # shape: (N_argo,)

    # If the minimal difference exceeds tol_ns, mark as -1
    idx[diff.min(axis=1) > tol_ns] = -1

    return idx


# ------------------------------------------------------------------
# Main loop: process each monthly ARGO file and produce a CSV
# ------------------------------------------------------------------
for argo_path in sorted(ARGO_DIR.glob("EN.4.2.2.f.profiles.g10.*.nc")):
    # Extract year and month from filename (format: EN.4.2.2.f.profiles.g10.YYYYMM.nc)
    m = re.search(r"(\d{4})(\d{2})", argo_path.name)
    if not m:
        continue
    year, month = int(m.group(1)), int(m.group(2))
    print(f"\nProcessing ARGO → SST for {year}-{month:02d}")

    # 1) Open the ARGO dataset for this month
    ds_a = open_argo_dataset(argo_path)
    if ds_a is None:
        continue

    # 2) Extract surface‐cast DataFrame for (year, month)
    df = process_argo(ds_a, year, month)
    ds_a.close()
    if df.empty:
        print("  No ARGO casts in this month → skipping")
        continue

    # 3) Time‐match each ARGO cast to the nearest SST snapshot index (within ±TIME_TOL_HOURS)
    df["SST_idx"] = vectorized_time_match(df["Time"], sst_times)
    df = df[df["SST_idx"] >= 0]  # keep only casts with a valid SST match
    if df.empty:
        print(f"  No ARGO→SST time‐matches (within ±{TIME_TOL_HOURS}h) → skipping")
        continue
    print(f"  {len(df)} ARGO casts matched to SST times")

    # 4) Spatially interpolate SST for each unique matched time index
    # ----------------------------------------------------------------
    df["SST (°C)"] = np.nan
    for t in df["SST_idx"].unique():
        if t < 0:
            continue

        # Extract the 2D SST snapshot at time‐index t (shape = [lat, lon])
        slc = ds_sst["analysed_sst"].isel(time=int(t)).values

        # Build a RegularGridInterpolator over (sst_lats, sst_lons)
        interp = RegularGridInterpolator(
            (sst_lats, sst_lons),
            slc,
            bounds_error=False,
            fill_value=np.nan
        )

        # For all ARGO casts whose SST_idx == t, interpolate SST at their lat/lon
        mask = (df["SST_idx"] == t)
        pts  = np.column_stack((df.loc[mask, "ARGO Lat"], df.loc[mask, "ARGO Lon"]))
        df.loc[mask, "SST (°C)"] = interp(pts)

    # 5) Compute ARGO – SST difference
    df["Difference (°C)"] = df["Temp ARGO (°C)"] - df["SST (°C)"]

    # 6) Drop helper columns and reorder for final CSV
    out = df.drop(columns=["Time", "SST_idx"])
    out = out[
        [
            "Day",
            "ARGO Lat",
            "ARGO Lon",
            "Temp ARGO (°C)",
            "SST (°C)",
            "Difference (°C)"
        ]
    ]

    # 7) Write the flat CSV for this month
    fn = ARGO_SST_FLAT_DIR / f"argo_sst_cmp_{year}{month:02d}.csv"
    out.to_csv(fn, index=False)
    print(f"  Wrote CSV → {fn.name}")

# Cleanup: close the Copernicus SST dataset
ds_sst.close()
