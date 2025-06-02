#!/usr/bin/env python3
"""
argo_sst_heatmaps.py

Generate monthly heatmaps of ARGO – Copernicus SST surface‐temperature differences.

For each ARGO‐SST comparison CSV (produced by argo_sst_compare.py), this script:
  1. Loads the CSV, which contains one row per ARGO surface cast:
       - Day, ARGO Lat, ARGO Lon, Temp ARGO (°C), SST (°C), Difference (°C).
  2. Filters out invalid or zero‐value measurements.
  3. Computes the mean ARGO–SST difference at each unique ARGO location (lat, lon).
  4. Interpolates those pointwise differences onto a regular lat/lon grid.
  5. Masks out any grid cells that fall on land.
  6. Produces a Plate Carrée contour‐filled map of the difference (±4 °C), with ARGO locations overplotted.
  7. Saves each monthly heatmap as a high‐resolution PNG under HEATMAP_BASE_DIR.

All directory paths and configuration values are imported from config.py:
  - ARGO_SST_FLAT_DIR: folder containing the monthly ARGO–SST CSVs.
  - HEATMAP_BASE_DIR:   base folder for writing output heatmap PNGs.
  - LAT_RANGE, LON_RANGE: geographic bounds for the interpolation and plot.

Usage:
    python argo_sst_heatmaps.py

Dependencies:
    numpy, pandas, matplotlib, scipy, cartopy, shapely, config.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep
from pathlib import Path
import calendar

import config

# Colormap for the difference plot
COLORMAP = 'RdBu_r'


# ------------------------------------------------------------------
# Paths (imported from config.py)
# ------------------------------------------------------------------
# Directory containing the “flat” ARGO–SST comparison CSVs
ARGO_SST_FLAT_DIR = config.OUTPUT_BASE / "argo_sst_flat"

# Base directory for writing heatmaps (will create a subfolder "argo_sst" here)
HEATMAP_BASE_DIR  = config.HEATMAP_BASE_DIR / "argo_sst"
HEATMAP_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Geographic bounds for the heatmap (latitude, longitude)
# You can adjust or expose these in config.py if needed
LAT_RANGE = (45, 82)
LON_RANGE = (-30, 30)


# ------------------------------------------------------------------
# Utility: Load one monthly CSV into a DataFrame
# ------------------------------------------------------------------
def load_comparison_data(csv_file: Path) -> pd.DataFrame:
    """
    Load a CSV containing ARGO–SST comparison for a single month.

    The CSV is expected to have columns:
      - 'Day' (date of cast)
      - 'ARGO Lat', 'ARGO Lon'
      - 'Temp ARGO (°C)', 'SST (°C)', 'Difference (°C)'

    Parameters
    ----------
    csv_file : Path
        Full path to the CSV file (e.g. "argo_sst_cmp_201901.csv").

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame. If empty, no rows are returned.
    """
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows from {csv_file.name}")
    return df


# ------------------------------------------------------------------
# Build a “land mask” for a given lat/lon grid
# ------------------------------------------------------------------
def get_land_mask(lat_grid: np.ndarray,
                  lon_grid: np.ndarray,
                  prepared_land_geoms):
    """
    Determine which grid points fall on land by checking against prepared land polygons.

    Parameters
    ----------
    lat_grid : 2D numpy array
        Latitude values of each grid cell (same shape as lon_grid).
    lon_grid : 2D numpy array
        Longitude values of each grid cell (same shape as lat_grid).
    prepared_land_geoms : list of shapely.prepared.PreparedGeometry
        A pre‐prepared list of land geometries, used for fast .contains() checks.

    Returns
    -------
    np.ndarray
        Boolean mask (same shape as lat_grid), where True indicates the cell is over land.
    """
    # Flatten lat/lon into point pairs
    pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    mask = [any(g.contains(Point(pt)) for g in prepared_land_geoms) for pt in pts]
    return np.array(mask).reshape(lat_grid.shape)


# ------------------------------------------------------------------
# Interpolate scattered differences onto a regular grid, then mask land
# ------------------------------------------------------------------
def interpolate_for_heatmap(lats: np.ndarray,
                            lons: np.ndarray,
                            values: np.ndarray,
                            resolution: int = 200,
                            lat_range=None,
                            lon_range=None,
                            prepared_land_geoms=None):
    """
    Interpolate scattered (latitude, longitude, difference) points onto a regular grid via linear interpolation.
    Then mask out grid cells that lie on land.

    Parameters
    ----------
    lats : 1D numpy array
        Latitudes of each data point (unique ARGO locations).
    lons : 1D numpy array
        Longitudes of each data point.
    values : 1D numpy array
        The ARGO–SST difference at each (lat, lon).
    resolution : int, optional
        Number of grid points along each axis (so the grid is resolution x resolution). Default is 200.
    lat_range : tuple (lat_min, lat_max), optional
        If provided, overrides the bounding box in latitude. Otherwise uses min(lats), max(lats).
    lon_range : tuple (lon_min, lon_max), optional
        If provided, overrides the bounding box in longitude.
    prepared_land_geoms : list of shapely.prepared.PreparedGeometry, optional
        Pre‐prepared land polygons. If not provided, land will not be masked.

    Returns
    -------
    lon_grid : 2D numpy array
        Regular longitude grid (shape = [resolution, resolution]).
    lat_grid : 2D numpy array
        Regular latitude grid.
    grid_vals : 2D masked array
        Interpolated difference values. Land cells are set to NaN.
    """
    # Determine bounding box from provided ranges or from data
    lat_min, lat_max = (lats.min(), lats.max()) if lat_range is None else lat_range
    lon_min, lon_max = (lons.min(), lons.max()) if lon_range is None else lon_range

    # Build a regular mesh of lat/lon
    lat_grid = np.linspace(lat_min, lat_max, resolution)
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Perform linear interpolation onto the grid
    grid_vals = griddata((lats, lons), values, (lat_grid, lon_grid), method='linear')
    grid_vals = np.ma.masked_invalid(grid_vals)

    # Mask out land if geometries are provided
    if prepared_land_geoms is not None:
        land_mask = get_land_mask(lat_grid, lon_grid, prepared_land_geoms)
        grid_vals = np.where(land_mask, np.nan, grid_vals)

    return lon_grid, lat_grid, grid_vals


# ------------------------------------------------------------------
# Plot one month’s ARGO–SST difference heatmap
# ------------------------------------------------------------------
def plot_heatmap_sst(lon_grid: np.ndarray,
                     lat_grid: np.ndarray,
                     diff_grid: np.ndarray,
                     month_name: str,
                     year: int,
                     lon_pts: np.ndarray,
                     lat_pts: np.ndarray,
                     output_path: Path):
    """
    Generate and save a filled‐contour map of ARGO – SST temperature difference.

    Parameters
    ----------
    lon_grid : 2D numpy array
        Longitude values of the interpolation grid.
    lat_grid : 2D numpy array
        Latitude values of the interpolation grid.
    diff_grid : 2D numpy array
        Interpolated difference values (ARGO – SST) on the grid (land as NaN).
    month_name : str
        Full month name (e.g., "January").
    year : int
        Four‐digit year (e.g., 2019).
    lon_pts : 1D numpy array
        Longitudes of ARGO cast locations (to overplot points).
    lat_pts : 1D numpy array
        Latitudes of ARGO cast locations.
    output_path : Path
        Full path (including filename) where the PNG will be saved.

    Notes
    -----
    - The color scale is clipped to [-4, +4] °C.
    - ARGO locations are plotted as white circles with black edge.
    - Coastlines, country borders, land (light gray) and ocean (light blue) are drawn.
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Contour levels and clipping
    vmin, vmax = -4, 4
    clipped = np.clip(diff_grid, vmin, vmax)

    cf = ax.contourf(
        lon_grid, lat_grid, clipped,
        levels=np.linspace(vmin, vmax, 100),
        cmap=COLORMAP,
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        extend='both'
    )

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label("Temp Difference (ARGO – SST) [°C]")
    cbar.set_ticks(np.arange(vmin, vmax + 1, 1))

    # Map features
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')

    # Scatter ARGO locations
    ax.scatter(
        lon_pts, lat_pts,
        c='white', s=10, edgecolor='k',
        transform=ccrs.PlateCarree(),
        label='ARGO casts'
    )

    # Title and labels
    ax.set_title(f"ARGO – SST Surface Temp Difference\n{month_name} {year}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {output_path.name}")


# ------------------------------------------------------------------
# Process one CSV and produce its heatmap
# ------------------------------------------------------------------
def create_monthly_heatmap_sst(csv_file: Path,
                               lat_range: tuple,
                               lon_range: tuple,
                               prepared_land_geoms,
                               output_dir: Path):
    """
    Load a single ARGO–SST comparison CSV, compute and plot the monthly average difference.

    Steps:
      1. Load the CSV into a DataFrame.
      2. Drop rows with NaN in required columns ('ARGO Lat', 'ARGO Lon',
         'Temp ARGO (°C)', 'SST (°C)'). Also drop any zero‐value
         measurements (invalid).
      3. Compute 'Difference (°C)' = 'Temp ARGO (°C)' – 'SST (°C)'.
      4. Group by (lat, lon) and compute mean difference.
      5. Interpolate these mean differences onto a regular grid via interpolate_for_heatmap.
      6. Determine year/month from filename to build title.
      7. Call plot_heatmap_sst to generate and save the figure.

    Parameters
    ----------
    csv_file : Path
        Path to the monthly CSV (e.g., "argo_sst_cmp_201901.csv").
    lat_range : tuple (lat_min, lat_max)
        Latitude bounds for interpolation (e.g., (45, 82)).
    lon_range : tuple (lon_min, lon_max)
        Longitude bounds for interpolation (e.g., (-30, 30)).
    prepared_land_geoms : list of shapely.prepared.PreparedGeometry
        Pre‐prepared land polygons for fast land‐masking.
    output_dir : Path
        Directory where the output PNG will be saved.
    """
    # Load the CSV
    df = load_comparison_data(csv_file)
    if df.empty:
        print("  → no data, skipping")
        return

    # Drop rows with NaN in required columns
    df = df.dropna(subset=['ARGO Lat', 'ARGO Lon', 'Temp ARGO (°C)', 'SST (°C)'])

    # Drop any cast with zero temp in either ARGO or SST (invalid)
    df = df[(df['Temp ARGO (°C)'] != 0) & (df['SST (°C)'] != 0)]
    if df.empty:
        print("  → no valid measurements, skipping")
        return

    # Compute the difference and group by (lat, lon) to get mean
    df['Difference (°C)'] = df['Temp ARGO (°C)'] - df['SST (°C)']
    grp = df.groupby(['ARGO Lat', 'ARGO Lon'], as_index=False)['Difference (°C)'].mean()

    # Extract arrays of lats, lons, and mean differences
    lats = grp['ARGO Lat'].values
    lons = grp['ARGO Lon'].values
    diffs = grp['Difference (°C)'].values

    # Interpolate to a regular grid
    lon_grid, lat_grid, diff_grid = interpolate_for_heatmap(
        lats, lons, diffs,
        resolution=200,
        lat_range=lat_range,
        lon_range=lon_range,
        prepared_land_geoms=prepared_land_geoms
    )

    # Parse year and month from CSV filename (e.g. "argo_sst_cmp_201901.csv" → 2019, 01)
    m = re.search(r"(\d{4})(\d{2})", csv_file.name)
    year, mon = int(m.group(1)), int(m.group(2))
    month_name = calendar.month_name[mon]

    # Build output path and plot
    output_path = output_dir / f"sst_cmp_argo_{year}{mon:02d}.png"
    plot_heatmap_sst(
        lon_grid, lat_grid, diff_grid,
        month_name, year,
        lons, lats,
        output_path
    )


# ------------------------------------------------------------------
# Main execution: loop over all CSVs in ARGO_SST_FLAT_DIR and make PNGs
# ------------------------------------------------------------------
if __name__ == "__main__":
    IN_DIR  = ARGO_SST_FLAT_DIR
    OUT_DIR = HEATMAP_BASE_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre‐prepare land polygons for masking (load once)
    land_geoms = list(cfeature.LAND.geometries())
    prepared_land = [prep(g) for g in land_geoms]

    # Find and sort all ARGO–SST CSV files
    files = sorted(IN_DIR.glob("argo_sst_cmp_*.csv"))
    print(f"Found {len(files)} ARGO–SST CSVs to map.")

    for csv_file in files:
        # Extract year/month from filename
        m = re.search(r"(\d{4})(\d{2})", csv_file.name)
        if not m:
            continue

        # If PNG already exists, skip
        out_png = OUT_DIR / f"sst_cmp_argo_{m.group(1)}{m.group(2)}.png"
        if out_png.exists():
            print(f"Skipping {csv_file.name} (already have {out_png.name})")
            continue

        print(f"\nProcessing {csv_file.name}")
        create_monthly_heatmap_sst(
            csv_file,
            lat_range=LAT_RANGE,
            lon_range=LON_RANGE,
            prepared_land_geoms=prepared_land,
            output_dir=OUT_DIR
        )
