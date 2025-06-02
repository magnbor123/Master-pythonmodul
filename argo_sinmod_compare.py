#!/usr/bin/env python3
"""
argo_sinmod_compare.py

Compare ARGO float profiles to SINMOD model output on a monthly basis,
for multiple SINMOD variants (“standard,” “nudged,” “assimilated”).

For each variant, year, and month:
  1. Load the ARGO NetCDF file for that month and extract all casts.
  2. Load the SINMOD NetCDF file corresponding to the chosen variant and year,
     and extract grid & time arrays for that month.
  3. For each ARGO cast:
       a. Restrict to casts that fall within the geographic domain of SINMOD.
       b. Match the ARGO cast’s date to the nearest SINMOD date within ±TIME_TOL_HOURS.
       c. Find the closest SINMOD grid cell in (x,y) to the cast’s (lat, lon).
       d. Extract the full ARGO profile (temperature & salinity vs. depth) at that cast.
       e. Interpolate the SINMOD 3D temperature and salinity to the ARGO depths.
       f. Record ARGO vs. SINMOD temperature & salinity at each depth.
  4. Save all ARGO–SINMOD depth‐by‐depth comparisons into a CSV.

Requires config.py in the same directory, which defines:
  - ARGO_DIR: Path to ARGO NetCDF files
  - get_sinmod_file(variant, year): function returning SINMOD file path
  - OUTPUT_DIRS: dict mapping each variant to its output folder for CSVs
  - YEARS, MONTHS: lists of years and months to process
  - TIME_TOL_HOURS: tolerance (hours) for matching ARGO dates to SINMOD dates
  - MAX_DIST_DEG: maximum distance (degrees) for matching cast location to SINMOD grid cell
"""

import time
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
from pathlib import Path

from config import (
    ARGO_DIR,             # Directory containing ARGO NetCDF files
    get_sinmod_file,      # Function to select SINMOD file given variant & year
    OUTPUT_DIRS,          # Dict: variant → output folder for CSVs
    YEARS,                # List of years to process (e.g. [2019, 2022, 2023])
    MONTHS,               # List of months to process (1–12)
    TIME_TOL_HOURS,       # Max hours tolerance when matching dates
    MAX_DIST_DEG          # Max degrees tolerance when matching lat/lon to SINMOD grid
)

def open_dataset(path: Path) -> xr.Dataset:
    """
    Open a NetCDF dataset with xarray, trying multiple engines in order:
    1) 'h5netcdf'
    2) 'netcdf4'
    If both fail, fall back to 'scipy' and immediately load everything into memory.

    This ensures that any .where(..., drop=True) calls work reliably.

    Parameters
    ----------
    path : Path
        Full path to the NetCDF file.

    Returns
    -------
    xr.Dataset
        The opened xarray dataset (lazy if using 'h5netcdf' or 'netcdf4', fully loaded if 'scipy').

    Raises
    ------
    Exception
        If none of the engines can open the file, the exception from the last attempt is propagated.
    """
    # First attempt: lazy loading via h5netcdf or netcdf4.
    for engine in ("h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception:
            pass

    # Last resort: use scipy engine and load entire dataset into memory immediately.
    ds = xr.open_dataset(path, engine="scipy")
    return ds.load()

def process_argo(ds: xr.Dataset, year: int, month: int):
    """
    Extract all ARGO floats cast in the specified year/month.

    Parameters
    ----------
    ds : xr.Dataset
        The ARGO NetCDF dataset for a given month (contains variables 'JULD', 'LATITUDE', 'LONGITUDE', 'TEMP', 'PSAL_CORRECTED', etc.).
    year : int
        Year of interest (e.g. 2019).
    month : int
        Month of interest (1–12).

    Returns
    -------
    (np.ndarray, np.ndarray, pd.DatetimeIndex, xr.Dataset)
        - argo_lats : 1D numpy array of latitudes of casts in that month.
        - argo_lons : 1D numpy array of longitudes of casts in that month.
        - argo_times: pandas.DatetimeIndex of cast times in that month.
        - ds         : The original xarray Dataset (returned for profiling later).
    
    Notes
    -----
    - If there are no casts in that year/month, returns three empty arrays and the original ds.
    """
    # Convert ARGO time variable 'JULD' to pandas timestamps
    times = pd.to_datetime(ds["JULD"].values)

    # Build a boolean mask for the desired year/month
    mask = (times.year == year) & (times.month == month)

    # If no casts in that month, return empty arrays
    if not mask.any():
        return np.empty(0), np.empty(0), pd.DatetimeIndex([]), ds

    # Extract lat/lon/times of casts that pass the mask
    argo_lats  = ds["LATITUDE"].values[mask]
    argo_lons  = ds["LONGITUDE"].values[mask]
    argo_times = times[mask]

    return argo_lats, argo_lons, argo_times, ds

def process_sinmod(ds: xr.Dataset, year: int, month: int):
    """
    Extract SINMOD grid and time arrays for the specified year/month.

    Parameters
    ----------
    ds : xr.Dataset
        Full SINMOD dataset (contains variables 'xc','yc','zc','time', etc.).
    year : int
        Year of interest (e.g. 2019).
    month : int
        Month of interest (1–12).

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, xr.Dataset)
        - sin_x  : 1D array of SINMOD x‐coordinates.
        - sin_y  : 1D array of SINMOD y‐coordinates.
        - sin_z  : 1D array of vertical levels (depths).
        - sin_t  : pandas.DatetimeIndex of SINMOD timestamps for that year/month.
        - ds     : The original xarray Dataset (returned for interpolation later).

    Notes
    -----
    - SINMOD’s time variable is stored as a 2D array of shape (N_time, 6), representing
      (year, month, day, hour, minute, second). We convert it into a single DatetimeIndex.
    - We then filter that DatetimeIndex to only those entries matching (year, month).
    """
    # Extract horizontal grid coordinates
    sin_x = ds["xc"].values
    sin_y = ds["yc"].values

    # Extract vertical levels (depths)
    sin_z = ds["zc"].values

    # Read the multi‐component time array of shape (N,6)
    t_raw = ds["time"].values  # e.g. [[2019,1,1,0,0,0], [2019,1,1,6,0,0], ...]

    # Combine into pandas datetime
    t_full = pd.to_datetime({
        "year":   t_raw[:, 0],
        "month":  t_raw[:, 1],
        "day":    t_raw[:, 2],
        "hour":   t_raw[:, 3],
        "minute": t_raw[:, 4],
        "second": t_raw[:, 5]
    })
    t_full = pd.DatetimeIndex(t_full)

    # Filter to only timestamps in the requested month
    mask = (t_full.year == year) & (t_full.month == month)

    return sin_x, sin_y, sin_z, t_full[mask], ds

def get_sinmod_proj(ds: xr.Dataset) -> Proj:
    """
    Build a polar‐stereographic pyproj.Proj object for the SINMOD grid.

    Parameters
    ----------
    ds : xr.Dataset
        A SINMOD xarray Dataset containing either a 'grid_mapping' variable or global mapping attributes.

    Returns
    -------
    Proj
        A pyproj.Proj instance configured for SINMOD’s polar‐stereographic projection.

    Notes
    -----
    - If the variable 'grid_mapping' exists and has non‐empty .attrs, use those attributes.
      Otherwise, fall back on global attributes (ds.attrs).
    - Default values are provided if some attributes are missing.
    """
    # Attempt to read attributes from 'grid_mapping' variable; if not present (KeyError) or empty,
    # fall back to global ds.attrs.
    try:
        gm = ds["grid_mapping"].attrs or ds.attrs
    except KeyError:
        gm = ds.attrs

    # Build the Proj object; use defaults if attributes are missing
    return Proj(
        proj="stere",
        lat_0=gm.get("latitude_of_projection_origin",       90.0),
        lon_0=gm.get("straight_vertical_longitude_from_pole", 58.0),
        lat_ts=gm.get("standard_parallel",                    60.0),
        x_0=gm.get("false_easting",                    3304000.0),
        y_0=gm.get("false_northing",                   2554000.0),
        a=  gm.get("semi_major_axis",                  6370000.0),
        b=  gm.get("semi_minor_axis",                  6370000.0),
    )

def compute_differences(
    ds_a: xr.Dataset, ds_s: xr.Dataset,
    argo_lats: np.ndarray, argo_lons: np.ndarray, argo_times: pd.DatetimeIndex,
    sin_x: np.ndarray, sin_y: np.ndarray, sin_z: np.ndarray, sin_t: pd.DatetimeIndex,
    proj: Proj
) -> pd.DataFrame:
    """
    Compute depth‐by‐depth ARGO vs. SINMOD differences for all casts in a given month.

    For each day in ARGO:
      1) Restrict ARGO casts to those within SINMOD’s horizontal domain.
      2) Group casts by calendar day.
      3) For each day:
         a) Find the nearest SINMOD timestamp to that day (within ±TIME_TOL_HOURS).
         b) Extract that day’s SINMOD 3D temperature & salinity arrays.
         c) For each ARGO cast on that day:
              i)   Find the closest grid cell (ix, iy) in SINMOD to the cast’s (lat, lon).
              ii)  Extract full ARGO profile (depth, T, S).
              iii) Interpolate SINMOD’s T(z) and S(z) onto each ARGO depth.
              iv)  Record: day, ARGO lat/lon/depth, SINMOD lat/lon, ARGO T/S, SINMOD T/S.

    Parameters
    ----------
    ds_a : xr.Dataset
        ARGO dataset for the month (used for slicing profiles).
    ds_s : xr.Dataset
        SINMOD dataset (the full file) for the variant/year.
    argo_lats : np.ndarray
        1D array of ARGO cast latitudes (only those in the month).
    argo_lons : np.ndarray
        1D array of ARGO cast longitudes.
    argo_times : pd.DatetimeIndex
        1D array of ARGO cast times (timestamps).
    sin_x : np.ndarray
        1D array of SINMOD grid x‐coordinates.
    sin_y : np.ndarray
        1D array of SINMOD grid y‐coordinates.
    sin_z : np.ndarray
        1D array of SINMOD vertical levels (depths).
    sin_t : pd.DatetimeIndex
        1D array of SINMOD timestamps (filtered to the month).
    proj : Proj
        A pyproj.Proj object for converting SINMOD’s (x,y) → (lon,lat).

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one (ARGO depth) measurement and contains:
          - Day (date of cast, floored to midnight),
          - ARGO Lat, ARGO Lon,
          - ARGO Depth,
          - SINMOD Lat, SINMOD Lon (closest grid cell),
          - Temp ARGO (°C), Temp SINMOD (°C),
          - Salinity ARGO (PSU), Salinity SINMOD (PSU).
          If any casts are skipped (out of domain, missing data), they are logged to 'skipped_argo_points.csv'.

    Notes
    -----
    - We build the “geographic” arrays SLON, SLAT once (meshgrid of sin_x, sin_y, converted via proj).
    - ARGO casts whose (lat,lon) fall outside the bounding box of (SLAT.min(), SLAT.max()) × (SLON.min(), SLON.max()) are skipped.
    - We group ARGO casts by day (floor the timestamp to midnight), then pick the nearest SINMOD date per day.
    - Horizontal matching: we carve a small (MAX_DIST_DEG)‐sided bounding box around each cast and pick the grid cell of minimum squared distance.
    - Vertical matching: simple 1D interpolation of SINMOD’s T(z), S(z) onto ARGO depths using np.interp.
    """
    rows  = []
    skips = []

    # Precompute geographic SINMOD grid (only once per month)
    SX, SY = np.meshgrid(sin_x, sin_y)          # 2D arrays of shape (len(sin_y), len(sin_x))
    SLON, SLAT = proj(SX, SY, inverse=True)     # Convert projected (x,y) → geographic (lon,lat)

    # Restrict ARGO casts to those inside SINMOD’s geographic domain
    dom_mask = (
        (argo_lats >= SLAT.min()) & (argo_lats <= SLAT.max()) &
        (argo_lons >= SLON.min()) & (argo_lons <= SLON.max())
    )
    argo_lats  = argo_lats[dom_mask]
    argo_lons  = argo_lons[dom_mask]
    argo_times = argo_times[dom_mask]
    print(f"Filtered ARGO → SINMOD domain: {len(argo_lats)} casts remain")

    # Group ARGO casts by calendar day
    days = pd.DatetimeIndex(argo_times.floor("D")).unique()

    for day in days:
        print(f"Processing day {day}")
        t0 = time.time()

        # Select all casts that occurred on this calendar day
        dmask     = argo_times.floor("D") == day
        cast_lats = argo_lats[dmask]
        cast_lons = argo_lons[dmask]
        if not len(cast_lats):
            continue

        # Find the SINMOD timestamp nearest to this day (within TIME_TOL_HOURS)
        diffs = np.abs(sin_t - day)
        if diffs.min() > pd.Timedelta(hours=TIME_TOL_HOURS):
            print(f"  → No SINMOD time within ±{TIME_TOL_HOURS}h; skipping day")
            continue
        it = diffs.argmin()

        # Extract SINMOD’s 3D temperature & salinity at that timestamp:
        # shape of T_sin, S_sin is (len(sin_z), len(sin_y), len(sin_x))
        T_sin = ds_s["temperature"].isel(time=it).values
        S_sin = ds_s["salinity"].isel(time=it).values

        # Loop through each cast on this day
        for la, lo in zip(cast_lats, cast_lons):
            # Build a tiny bounding box around (la, lo) of ± MAX_DIST_DEG
            lat_mask = (SLAT >= la - MAX_DIST_DEG) & (SLAT <= la + MAX_DIST_DEG)
            lon_mask = (SLON >= lo - MAX_DIST_DEG) & (SLON <= lo + MAX_DIST_DEG)
            box      = lat_mask & lon_mask

            # If no SINMOD grid cells lie in that bounding box, skip this cast
            if not box.any():
                skips.append((la, lo, "out of bbox"))
                continue

            # Among the candidates in the bounding box, pick the cell of minimum squared distance
            yy, xx = np.where(box)
            d2     = (SLAT[yy, xx] - la)**2 + (SLON[yy, xx] - lo)**2
            j      = d2.argmin()
            iy, ix = int(yy[j]), int(xx[j])  # indices of the closest SINMOD grid cell

            # Extract the ARGO profile at exactly (lat, lon) = (la, lo)
            prof   = ds_a.where(
                (ds_a["LATITUDE"] == la) & (ds_a["LONGITUDE"] == lo),
                drop=True
            )
            depths = prof["DEPH_CORRECTED"].values.flatten()  # 1D depths array
            targo  = prof["TEMP"].values.flatten()            # ARGO temperatures at each depth
            sargo  = prof["PSAL_CORRECTED"].values.flatten()  # ARGO salinities at each depth

            # If ARGO profile is missing, skip
            if depths.size == 0:
                skips.append((la, lo, "missing profile"))
                continue

            # Interpolate SINMOD’s 3D T(z) and S(z) onto each ARGO depth
            ts_all = np.interp(depths, sin_z, T_sin[:, iy, ix])
            ss_all = np.interp(depths, sin_z, S_sin[:, iy, ix])

            # For each depth in the ARGO profile, record ARGO vs. SINMOD values
            for dep, ta, sa, ts, ss in zip(depths, targo, sargo, ts_all, ss_all):
                rows.append({
                    "Day":                   day,
                    "ARGO Lat":              float(la),
                    "ARGO Lon":              float(lo),
                    "ARGO Depth":            float(dep),
                    "SINMOD Lat":            float(SLAT[iy, ix]),
                    "SINMOD Lon":            float(SLON[iy, ix]),
                    "Temp ARGO (°C)":        float(ta),
                    "Temp SINMOD (°C)":      float(ts),
                    "Salinity ARGO (PSU)":   float(sa),
                    "Salinity SINMOD (PSU)": float(ss),
                })

        print(f"  Day processed in {time.time() - t0:.1f}s")

    # If any casts were skipped, log them to a CSV for later inspection
    if skips:
        skip_df = pd.DataFrame(skips, columns=("Lat", "Lon", "reason"))
        skip_df.to_csv("skipped_argo_points.csv", index=False)
        print(f"Skipped {len(skips)} casts; details written to skipped_argo_points.csv")

    return pd.DataFrame(rows)

def main():
    """
    Main driver function: loops over each SINMOD variant and each (year, month),
    invokes the ARGO→SINMOD comparison, and writes results to CSV.
    """
    for variant, out_dir in OUTPUT_DIRS.items():
        print(f"\n=== Variant: {variant} → using SINMOD file: {get_sinmod_file(variant, YEARS[0]).name} ===")
        out_dir.mkdir(parents=True, exist_ok=True)

        for year in YEARS:
            for month in MONTHS:
                # Build Paths for ARGO and SINMOD files
                argo_fp   = ARGO_DIR / f"EN.4.2.2.f.profiles.g10.{year}{month:02d}.nc"
                sinmod_fp = get_sinmod_file(variant, year)
                out_csv   = out_dir / f"{variant}_{year}{month:02d}.csv"

                print(f"-- Processing {year}-{month:02d} → will write {out_csv.name}")

                # If ARGO file does not exist for that month, skip
                if not argo_fp.exists():
                    print("   ARGO file missing; skipping.")
                    continue

                # Load ARGO and SINMOD datasets
                ds_a = open_dataset(argo_fp)
                ds_s = open_dataset(sinmod_fp)

                # Extract ARGO casts and SINMOD grid/time for that month
                alats, alons, atimes, ds_a = process_argo(ds_a, year, month)
                sx, sy, sz, stimes, ds_s = process_sinmod(ds_s, year, month)

                # If either no ARGO casts or no SINMOD times in that month, skip
                if alats.size == 0 or stimes.empty:
                    print("   No data for this month; skipping.")
                    ds_a.close()
                    ds_s.close()
                    continue

                # Build projection for SINMOD coordinates
                proj = get_sinmod_proj(ds_s)

                # Compute the depth‐by‐depth differences and gather into a DataFrame
                df = compute_differences(
                    ds_a, ds_s,
                    alats, alons, atimes,
                    sx, sy, sz, stimes,
                    proj
                )

                # Write the comparison table to CSV
                df.to_csv(out_csv, index=False)
                print(f"   Wrote CSV: {out_csv}")

                # Close datasets to free memory
                ds_a.close()
                ds_s.close()

if __name__ == "__main__":
    main()
