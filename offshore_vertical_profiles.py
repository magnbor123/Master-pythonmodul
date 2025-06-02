#!/usr/bin/env python3
"""
offshore_vertical_profiles.py

Generate combined vertical profiles (temperature and salinity vs. depth) for ARGO casts
that lie offshore (beyond a specified distance from the Norwegian and Svalbard coast),
comparing ARGO measurements to multiple SINMOD model variants (standard, nudged, assimilated).
For each month and each qualifying offshore cast:

  1. Identify ARGO casts with depth ≥ VERT_DEPTH_MIN and farther than VERT_COAST_KM from the coast.
  2. Ensure the cast appears in all available SINMOD‐comparison CSVs (one per variant).
  3. Merge ARGO and SINMOD model values for that cast and depth range.
  4. Determine the shallowest maximum depth common to all available model variants.
  5. Load the nearest Copernicus SST for the cast date/location.
  6. Plot ARGO, standard, nudged, and assimilated temperature and salinity profiles
     on a dual‐axis figure, marking the SST at depth = 0.
  7. Save each plot to a structured directory under PROFILE_BASE.

Configuration is imported from config.py:
  - OUTPUT_DIRS:         dict mapping each SINMOD variant → its CSV directory
  - YEARS, MONTHS:       lists of years and months to process
  - VERT_DEPTH_MIN:      minimum ARGO depth (m) for inclusion (overridden to 500 m here)
  - VERT_COAST_KM:       minimum distance (km) from coast for “offshore” casts
  - VERT_PROFILE_BASE:   base directory for saving profile plots
  - SST_DIR:             directory containing Copernicus SST NetCDF files (used to fetch an SST value)
  - MAX_PROFILES_PER_MONTH: maximum number of offshore profiles to plot per month

Dependencies:
    pandas, numpy, matplotlib, xarray, shapely, cartopy, config.py
"""

import glob
import random
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
OUTPUT_DIRS      = config.OUTPUT_DIRS            # {variant: path/to/csvs}
YEARS            = config.YEARS                  # e.g., [2019, 2022, 2023]
MONTHS           = config.MONTHS                 # [1, 2, ..., 12]
VERT_DEPTH_MIN   = 500                            # override: only include casts ≥ 500 m
VERT_COAST_KM    = config.VERT_COAST_KM          # e.g., 10.0 km
PROFILE_BASE     = config.VERT_PROFILE_BASE       # base directory for profile plots
SST_DIR          = config.SST_DIR                # Copernicus SST directory
MAX_PROFILES     = config.MAX_PROFILES_PER_MONTH  # how many offline profiles to plot per month


# ─── BUILD CLIPPED NORWAY+SVALBARD COASTLINE ──────────────────────────────────
# Load the Natural Earth coastline (50m resolution), clip to Norway+Svalbard box,
# and prepare a Shapely geometry for fast distance‐to‐coast queries.
coast_feat  = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
coast_union = unary_union(list(coast_feat.geometries()))
clip_box    = box(-10.0, 58.0, 40.0, 82.0)  # (lon_min, lat_min, lon_max, lat_max)
nor_coast   = coast_union.intersection(clip_box)
prep_coast  = prep(nor_coast)


def dist_to_coast_km(lon: float, lat: float) -> float:
    """
    Approximate the distance (in kilometers) from geographic point (lon, lat)
    to the clipped Norway+Svalbard coastline.

    Parameters
    ----------
    lon : float
        Longitude of the point (degrees).
    lat : float
        Latitude of the point (degrees).

    Returns
    -------
    float
        Distance (km) to nearest point on nor_coast. Uses 1° ≈ 111 km conversion.
    """
    # Shapely distance returns degrees; multiply by ~111 to convert to kilometers
    return nor_coast.distance(Point(lon, lat)) * 111.0


# ─── SST LOADER ────────────────────────────────────────────────────────────────
def load_sst_value(date: pd.Timestamp, lat: float, lon: float) -> float:
    """
    Retrieve the nearest Copernicus SST value (°C) at a given date and location.

    Procedure:
      1. Choose the appropriate Copernicus file by year and file‐naming convention:
         - If month ≤ 8, match "*jan*aug*.nc"; otherwise match "*sep*.nc".
      2. Load the first matching file, convert SST from Kelvin to Celsius.
      3. Find the time index closest to `date`.
      4. Extract the 2D SST slice (lat × lon), or if only 2D data, use directly.
      5. Find the nearest lat and lon indices, return SST at [i,j].

    Parameters
    ----------
    date : pd.Timestamp
        Target date/time for SST retrieval.
    lat : float
        Latitude for which to retrieve SST.
    lon : float
        Longitude for which to retrieve SST.

    Returns
    -------
    float or None
        SST (°C) at nearest time & location, or None if no file is found.
    """
    year, month = date.year, date.month
    # Determine file pattern based on month
    patt = "*jan*aug*.nc" if month <= 8 else "*sep*.nc"
    files = glob.glob(str(SST_DIR / f"*{year}*{patt}"))
    if not files:
        return None

    # Open the first matching Copernicus file
    ds = xr.open_dataset(files[0], engine="h5netcdf")
    # Convert Kelvin → Celsius
    sst_all = ds["analysed_sst"].values - 273.15
    times   = pd.to_datetime(ds["time"].values)

    # Find index of time closest to `date`
    idx_time = np.abs(times - date).argmin()
    # If SST data is 3D (time, lat, lon), we take the slice at idx_time; if 2D, use directly
    slice_ = sst_all[idx_time] if sst_all.ndim == 3 else sst_all

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()

    # Find nearest latitude and longitude indices
    i = np.abs(lats - lat).argmin()
    j = np.abs(lons - lon).argmin()
    return float(slice_[i, j])


# ─── KEY LOADER ────────────────────────────────────────────────────────────────
def load_keys(path: Path) -> pd.DataFrame:
    """
    Load only the key columns ("Day", "ARGO Lat", "ARGO Lon", "ARGO Depth")
    from a monthly ARGO–SINMOD comparison CSV, coerce datatypes, round values,
    and group duplicates by taking the maximum depth.

    Steps:
      1. Read 'Day' column as datetime, coerce errors → NaT.
      2. Round 'ARGO Lat' and 'ARGO Lon' to 0.001°, 'ARGO Depth' to 0.1 m.
      3. Group by (Day, ARGO Lat, ARGO Lon) and take the maximum ARGO Depth.

    Parameters
    ----------
    path : Path
        Path to the comparison CSV (e.g., "standard_201901.csv").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["Day", "ARGO Lat", "ARGO Lon", "ARGO Depth"] and
        one row per unique (Day, Lat, Lon), depth = max depth if duplicates exist.
    """
    df = pd.read_csv(path, usecols=["Day", "ARGO Lat", "ARGO Lon", "ARGO Depth"])
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

    # Round lat, lon, depth to reduce floating‐point noise
    df["ARGO Lat"]   = pd.to_numeric(df["ARGO Lat"], errors="coerce").round(3)
    df["ARGO Lon"]   = pd.to_numeric(df["ARGO Lon"], errors="coerce").round(3)
    df["ARGO Depth"] = pd.to_numeric(df["ARGO Depth"], errors="coerce").round(1)

    # Group by (Day, Lat, Lon) and take max depth
    return df.groupby(["Day", "ARGO Lat", "ARGO Lon"], as_index=False)["ARGO Depth"].max()


# ─── FULL PROFILE LOADER ───────────────────────────────────────────────────────
def load_and_rename(path: Path, sim: str) -> pd.DataFrame:
    """
    Load a full monthly ARGO–SINMOD comparison CSV for a given model variant and
    rename its SINMOD columns to include the variant name.

    Steps:
      1. Read the CSV into a DataFrame.
      2. Convert 'Day' to datetime, rounding numeric columns.
      3. Rename "Temp SINMOD (°C)" → "Temp {sim} (°C)" and
         "Salinity SINMOD (PSU)" → "Salinity {sim} (PSU)".

    Parameters
    ----------
    path : Path
        Path to the CSV for one variant (e.g., "nudged_201901.csv").
    sim : str
        Variant name, one of "standard", "nudged", or "assimilated".

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame with renamed SINMOD columns and rounded numeric values.
    """
    df = pd.read_csv(path)
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")

    # Build rename map based on variant
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

    # Round key numeric columns
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

    1. Sort the DataFrame `group` by 'ARGO Depth' ascending.
    2. Extract arrays for:
       - `depth`: ARGO Depth (1D).
       - `t_argo`, `s_argo`: ARGO temperature and salinity arrays.
       - `t_std`, `t_nud`, `t_ass`: SINMOD temperature arrays (variants or NaN).
       - `s_std`, `s_nud`, `s_ass`: SINMOD salinity arrays (variants or NaN).
    3. Create a figure with two subplots side by side:
       - Left (ax0): Temperature profile.
       - Right (ax1): Salinity profile.
    4. For each subplot:
       - Plot ARGO in black.
       - Plot Standard model in green (if present).
       - Plot Nudged model in blue (if present).
       - Plot Assimilated model in red (if present).
       - If `sst` is provided, scatter a yellow dot at (sst, depth=0).
       - Invert the y-axis so `depth=0` is at the top.
       - Add gridlines, labels, title, and legend.
    5. Set a super‐title with date, location, and max depth.
    6. Save the figure to `outpath`.

    Parameters
    ----------
    group : pd.DataFrame
        A DataFrame corresponding to one ARGO cast (unique Day, Lat, Lon), with columns:
          - 'ARGO Depth', 'Temp ARGO (°C)', 'Salinity ARGO (PSU)'
          - 'Temp standard (°C)', 'Temp nudged (°C)', 'Temp assimilated (°C)'
          - 'Salinity standard (PSU)', 'Salinity nudged (PSU)', 'Salinity assimilated (PSU)'
    outpath : Path
        Full path (including filename) where the PNG will be saved.
    sst : float or None, optional
        SST at the cast’s surface (°C). If provided, plotted as a yellow circle at depth = 0.
    """
    # Sort by depth ascending
    group = group.sort_values("ARGO Depth")
    depth  = group["ARGO Depth"].values
    t_argo = group["Temp ARGO (°C)"].values
    s_argo = group["Salinity ARGO (PSU)"].values

    # Extract SINMOD temperature and salinity arrays for each variant, or fill with NaN
    t_std = group.get("Temp standard (°C)",      np.full_like(depth, np.nan))
    t_nud = group.get("Temp nudged (°C)",        np.full_like(depth, np.nan))
    t_ass = group.get("Temp assimilated (°C)",   np.full_like(depth, np.nan))
    s_std = group.get("Salinity standard (PSU)", np.full_like(depth, np.nan))
    s_nud = group.get("Salinity nudged (PSU)",   np.full_like(depth, np.nan))
    s_ass = group.get("Salinity assimilated (PSU)", np.full_like(depth, np.nan))

    # Create a 1×2 subplot: left for temperature, right for salinity
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Temperature subplot (ax0) ---
    handles, labels = [], []
    # Plot ARGO in black
    h = ax0.plot(t_argo, depth, 'k-', label="ARGO")[0]
    handles.append(h); labels.append("ARGO")
    # Plot Standard model in green if data present
    if np.any(~np.isnan(t_std)):
        h = ax0.plot(t_std, depth, 'g-', label="Standard")[0]
        handles.append(h); labels.append("Standard")
    # Plot Nudged model in blue if data present
    if np.any(~np.isnan(t_nud)):
        h = ax0.plot(t_nud, depth, 'b-', label="Nudged")[0]
        handles.append(h); labels.append("Nudged")
    # Plot Assimilated model in red if data present
    if np.any(~np.isnan(t_ass)):
        h = ax0.plot(t_ass, depth, 'r-', label="Assimilated")[0]
        handles.append(h); labels.append("Assimilated")

    # If SST is provided, scatter a yellow dot at depth=0
    if sst is not None:
        h = ax0.scatter(
            [sst], [0],
            s=50, edgecolor='k',
            facecolor='yellow',
            label="SST", zorder=10
        )
        handles.append(h); labels.append("SST")

    ax0.invert_yaxis()  # so 0 m (surface) is at top
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

    # --- Super‐title with cast metadata ---
    day      = group["Day"].dt.strftime("%Y-%m-%d").iloc[0]
    lat, lon = group["ARGO Lat"].iloc[0], group["ARGO Lon"].iloc[0]
    dmax     = int(group["ARGO Depth"].max())
    fig.suptitle(
        f"Offshore Profile {day} ({lat:.3f},{lon:.3f}), max depth = {dmax} m",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    """
    Main driver: loop over each year/month, find offshore ARGO casts present in all variants,
    load their full profiles, merge ARGO & SINMOD data, fetch SST, and plot combined vertical profiles.

    Steps for each (year, month):
      1. Build CSV paths for "standard", "nudged", "assimilated" variants.
      2. Skip month if no CSV exists in any variant.
      3. Identify which variants have data this month.
      4. Load keys (Day, Lat, Lon, Depth) for each existing variant, require Depth ≥ VERT_DEPTH_MIN.
      5. Find the intersection of keys across all existing variants.
      6. Filter that intersection to only offshore casts (distance > VERT_COAST_KM).
      7. If more than MAX_PROFILES casts, randomly sample MAX_PROFILES.
      8. For each offshore cast:
         a. Merge full profile data from each existing variant (matching by Day, Lat, Lon, Depth).
         b. Ensure "standard" variant has at least one non‐NaN measurement.
         c. Determine the shallowest common maximum ARGO Depth among all variants.
         d. Truncate merged DataFrame to Depth ≤ common_max.
         e. Fetch SST from Copernicus for cast date/location.
         f. Call plot_combined_profile to generate and save the figure.
    """
    for year in YEARS:
        for month in MONTHS:
            ym = f"{year}{month:02d}"
            print(f"\n→ Processing {ym}")

            # Build CSV paths for each variant this month
            paths = {
                v: OUTPUT_DIRS[v] / f"{v}_{ym}.csv"
                for v in ("standard", "nudged", "assimilated")
            }

            # If no CSV exists for any variant, skip the month
            if not any(p.exists() for p in paths.values()):
                print("   No CSVs found for any variant; skipping month.")
                continue

            # Determine which variants actually have data
            existing_variants = [v for v, p in paths.items() if p.exists()]

            # Load keys for each existing variant, filter by minimum depth, collect sets of keys
            key_sets = []
            for v in existing_variants:
                dfk = load_keys(paths[v])
                dfk = dfk[dfk["ARGO Depth"] >= VERT_DEPTH_MIN]
                # Build set of unique keys (Day, Lat, Lon) for this variant
                keys = {
                    (row.Day.strftime("%Y-%m-%d"), row["ARGO Lat"], row["ARGO Lon"])
                    for _, row in dfk.iterrows()
                }
                key_sets.append(keys)

            # Find intersection (casts present in all variants)
            common_keys = set.intersection(*key_sets) if key_sets else set()
            if not common_keys:
                print("   No casts present in all variants; skipping month.")
                continue

            # Filter to only offshore casts (distance to coast > VERT_COAST_KM)
            offshore_keys = [
                key for key in common_keys
                if dist_to_coast_km(key[2], key[1]) > VERT_COAST_KM
            ]
            if not offshore_keys:
                print(f"   No offshore casts (beyond {VERT_COAST_KM} km); skipping month.")
                continue

            # If more than MAX_PROFILES, randomly sample
            if len(offshore_keys) > MAX_PROFILES:
                offshore_keys = random.sample(offshore_keys, MAX_PROFILES)
                print(f"   Sampling {MAX_PROFILES} of {len(offshore_keys)} offshore casts")

            # Prepare the output folder for this month and depth
            out_folder = PROFILE_BASE / f"depth_{int(VERT_DEPTH_MIN)}m" / "offshore" / ym
            out_folder.mkdir(parents=True, exist_ok=True)

            # Loop over each chosen offshore cast
            for day_str, lat, lon in offshore_keys:
                day_dt = pd.to_datetime(day_str)

                # Merge full profiles from each existing variant into a single DataFrame
                merged = None
                for v in existing_variants:
                    df = load_and_rename(paths[v], v)
                    # Add a helper column "key" to match Day, Lat, Lon
                    df["key"] = df.apply(
                        lambda r: (r["Day"].strftime("%Y-%m-%d"),
                                   r["ARGO Lat"], r["ARGO Lon"]),
                        axis=1
                    )
                    # Select only rows matching this cast
                    sub = df[df["key"] == (day_str, lat, lon)]
                    if sub.empty:
                        continue
                    # Merge on ARGO columns & Depth
                    if merged is None:
                        merged = sub.copy()
                    else:
                        merged = pd.merge(
                            merged, sub,
                            on=[
                                "Day", "ARGO Lat", "ARGO Lon", "ARGO Depth",
                                "Temp ARGO (°C)", "Salinity ARGO (PSU)"
                            ],
                            how="outer"
                        )

                # If no merged data was found, skip
                if merged is None:
                    continue

                # Ensure that the standard model has at least one non‐NaN value
                if merged["Temp standard (°C)"].isna().all() and \
                   merged["Salinity standard (PSU)"].isna().all():
                    continue

                # Determine the shallowest common max depth across available variants
                max_depths = []
                for v in ("standard", "nudged", "assimilated"):
                    col_name = f"Temp {v} (°C)" if v != "standard" else "Temp standard (°C)"
                    if col_name in merged.columns and merged[col_name].notna().any():
                        d = merged.loc[merged[col_name].notna(), "ARGO Depth"].max()
                        max_depths.append(d)
                common_max = min(max_depths)

                # Keep only rows where ARGO Depth ≤ common_max
                merged = merged[merged["ARGO Depth"] <= common_max]

                # Load the nearest Copernicus SST at (day_dt, lat, lon)
                sst = load_sst_value(day_dt, lat, lon)

                # Build output filename and plot
                fname = f"profile_{ym}_{day_str}_{lat:.3f}_{lon:.3f}.png"
                outpath = out_folder / fname
                plot_combined_profile(merged, outpath, sst)


if __name__ == "__main__":
    main()
