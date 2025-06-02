#!/usr/bin/env python3
"""
argo_sinmod_heatmaps.py

Generate monthly heatmaps of ARGO‐SINMOD differences (temperature or salinity) at a fixed depth.

For each SINMOD variant ("standard", "nudged", "assimilated"), this script:
  1. Reads the corresponding ARGO‐vs‐SINMOD CSV files (one per month) from HEATMAP_INPUT_DIRS.
  2. Filters the data for a given target depth ± tolerance, and within lat/lon bounds.
  3. Computes the mean ARGO – SINMOD difference at each unique (lat, lon) location.
  4. Interpolates those pointwise differences onto a regular grid.
  5. Optionally overlays bathymetry contours from the SINMOD file.
  6. Produces a contour‐filled map (with coastlines, land/ocean coloring, and scatter of ARGO points),
     saving each map as a PNG under HEATMAP_BASE_DIR.

All file‐ and directory‐paths, as well as SINMOD file locations, are imported from config.py:
  - HEATMAP_INPUT_DIRS: dict mapping each variant to the folder of CSV inputs
  - HEATMAP_BASE_DIR:    base directory for writing output PNGs
  - HEATMAP_SINMOD_FILE: dict mapping each variant to its SINMOD NetCDF file path

Usage
-----
    python argo_sinmod_heatmaps.py

You can adjust parameters (e.g. depth, tolerance, lat/lon range, resolution) in the `main()` function.
"""

import os
import glob
import calendar
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep
from pyproj import Transformer, Proj
import xarray as xr

# ─── SETTINGS ───────────────────────────────────────────────────────────────────
# Import directories and SINMOD file paths from config.py
from config import HEATMAP_INPUT_DIRS, HEATMAP_BASE_DIR, HEATMAP_SINMOD_FILE

# Colormap used for both temperature and salinity difference maps
colormap = 'RdYlBu_r'


# ─── DATA LOADING & INTERPOLATION ───────────────────────────────────────────────

def load_comparison_data(csv_file: str) -> pd.DataFrame:
    """
    Load the ARGO‐SINMOD comparison CSV for a given month & variant.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file (e.g., "standard_201901.csv").

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns:
          - 'ARGO Lat', 'ARGO Lon', 'ARGO Depth'
          - 'Temp ARGO (°C)', 'Temp SINMOD (°C)'
          - 'Salinity ARGO (PSU)', 'Salinity SINMOD (PSU)'
          - plus 'Day' (date of cast)
    """
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} rows from {csv_file}")
    return df


def get_sinmod_projection(ds: xr.Dataset) -> Proj:
    """
    Build a polar‐stereographic pyproj.Proj object from a SINMOD dataset.

    The function checks for a 'grid_mapping' variable. If not found or empty,
    it falls back to global attributes (ds.attrs).

    Parameters
    ----------
    ds : xr.Dataset
        SINMOD dataset (opened with xarray).

    Returns
    -------
    Proj
        A pyproj.Proj instance configured for SINMOD's projection.
    """
    try:
        gm = ds['grid_mapping'].attrs
    except KeyError:
        gm = ds.attrs

    return Proj(
        proj='stere',
        lat_0=gm.get('latitude_of_projection_origin', 90.0),
        lon_0=gm.get('straight_vertical_longitude_from_pole', 58.0),
        lat_ts=gm.get('standard_parallel', 60.0),
        x_0=gm.get('false_easting', 3304000.0),
        y_0=gm.get('false_northing', 2554000.0),
        a=  gm.get('semi_major_axis', 6370000.0),
        b=  gm.get('semi_minor_axis', 6370000.0)
    )


def get_land_mask(lat_grid: np.ndarray,
                  lon_grid: np.ndarray,
                  prepared_land_geoms=None) -> np.ndarray:
    """
    Determine which grid points fall on land, using Cartopy's built‐in land polygons.

    Parameters
    ----------
    lat_grid : 2D numpy array
        Latitude values of each grid cell (same shape as lon_grid).
    lon_grid : 2D numpy array
        Longitude values of each grid cell (same shape as lat_grid).
    prepared_land_geoms : list of Shapely geometries, optional
        Pre‐prepared land polygons (to speed up .contains() checks).
        If None, the function will retrieve and prepare Cartopy's LAND geometries.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (lat_grid.shape), where True indicates land.
    """
    if prepared_land_geoms is None:
        land_geoms = list(cfeature.LAND.geometries())
        prepared_land_geoms = [prep(g) for g in land_geoms]

    pts = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    mask = np.array([
        any(g.contains(Point(xy)) for g in prepared_land_geoms)
        for xy in pts
    ])
    return mask.reshape(lat_grid.shape)


def interpolate_for_heatmap(lats: np.ndarray,
                            lons: np.ndarray,
                            values: np.ndarray,
                            resolution: int = 100,
                            lat_range=None,
                            lon_range=None,
                            prepared_land_geoms=None):
    """
    Interpolate scattered (lat, lon, value) points onto a regular grid, then mask out land.

    Parameters
    ----------
    lats : 1D array
        Latitude of each data point.
    lons : 1D array
        Longitude of each data point.
    values : 1D array
        The ARGO–SINMOD difference values at each (lat, lon).
    resolution : int, optional
        Number of grid points along each axis (i.e. final grid is resolution × resolution).
        Default is 100.
    lat_range : tuple (lat_min, lat_max), optional
        If provided, defines the bounding box in latitude. Otherwise uses min/max of lats.
    lon_range : tuple (lon_min, lon_max), optional
        If provided, defines the bounding box in longitude.
    prepared_land_geoms : list of Shapely geometry, optional
        Pre‐prepared land polygons. If None, will prepare on the fly.

    Returns
    -------
    XI : 2D array (shape = [resolution, resolution])
        Longitude grid for plotting.
    YI : 2D array (shape = [resolution, resolution])
        Latitude grid for plotting.
    VI : 2D masked array
        Interpolated values on the grid, with land cells masked as NaN.
    """
    # Determine bounding box from provided ranges or from data
    lat_min, lat_max = (lats.min(), lats.max()) if lat_range is None else lat_range
    lon_min, lon_max = (lons.min(), lons.max()) if lon_range is None else lon_range

    # Build a regular lat/lon grid
    xi = np.linspace(lon_min, lon_max, resolution)
    yi = np.linspace(lat_min, lat_max, resolution)
    XI, YI = np.meshgrid(xi, yi)

    # Perform linear interpolation at each grid cell
    VI = griddata((lons, lats), values, (XI, YI), method='linear')
    VI = np.ma.masked_invalid(VI)  # mask out NaNs where interpolation failed

    # Mask out land areas
    land_mask = get_land_mask(YI, XI, prepared_land_geoms)
    VI[land_mask] = np.nan

    return XI, YI, VI


# ─── PLOTTING ────────────────────────────────────────────────────────────────────

def plot_heatmap(XI: np.ndarray,
                 YI: np.ndarray,
                 VI: np.ndarray,
                 depth: float,
                 parameter: str,
                 tolerance: float,
                 scatter_lats: np.ndarray,
                 scatter_lons: np.ndarray,
                 contour_depths,
                 output_path: Path):
    """
    Generate and save a contour‐filled heatmap of ARGO–SINMOD differences at a fixed depth.

    Parameters
    ----------
    XI : 2D array
        Regular longitude grid (from interpolate_for_heatmap).
    YI : 2D array
        Regular latitude grid.
    VI : 2D masked array
        Interpolated difference values (ARGO – SINMOD) on the grid, with land masked as NaN.
    depth : float
        Target depth (m) at which differences are computed.
    parameter : str
        Either "Temperature" or "Salinity". Determines color scale and labels.
    tolerance : float
        Depth tolerance (m) around the target depth (used in the title).
    scatter_lats : 1D array
        Latitudes of ARGO points used for overplotting.
    scatter_lons : 1D array
        Longitudes of ARGO points.
    contour_depths : 2D array or None
        Bathymetry (depth) grid, used to overlay isobaths if provided.
        Should be the same shape as XI, YI. If None, skip contouring.
    output_path : Path
        Path to save the final PNG.

    Notes
    -----
    - Temperature differences are clipped to [–4, +4] °C; salinity differences to [–1, +1] PSU.
    - ARGO points are plotted as white circles with black edges for reference.
    - If `contour_depths` is provided, draws major bathymetric contours (200, 500, 1000, 2000, 3000 m).
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Determine levels & clipping based on parameter
    if parameter == 'Temperature':
        levels = np.linspace(-4, 4, 100)
        clip = np.clip(VI, -4, 4)
    else:
        levels = np.linspace(-1, 1, 100)
        clip = np.clip(VI, -1, 1)

    # Filled contour
    cs = ax.contourf(
        XI, YI, clip,
        levels=levels,
        cmap=colormap,
        extend='both',
        transform=ccrs.PlateCarree()
    )

    # Colorbar
    cb = plt.colorbar(cs, label=f'{parameter} Difference (ARGO − SINMOD)')
    if parameter == 'Salinity':
        cb.set_ticks(np.linspace(-1, 1, 5))
    else:
        cb.set_ticks(np.arange(-4, 5, 1))

    # Map features
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Scatter ARGO points on top
    ax.scatter(
        scatter_lons, scatter_lats,
        s=20, edgecolor='black', facecolor='white',
        transform=ccrs.PlateCarree(), zorder=3,
        label='ARGO locations'
    )

    # Optionally overlay bathymetry contours
    if contour_depths is not None:
        major_isobaths = [200, 500, 1000, 2000, 3000]
        Cc = ax.contour(
            XI, YI, contour_depths,
            levels=major_isobaths,
            colors='gray',
            linewidths=0.75,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(Cc, inline=True, fontsize=8, fmt='%dm')

    # Title and labels
    ax.set_title(f'{parameter} Δ at {depth:.0f}m (±{tolerance}m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='lower left')

    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# ─── MONTHLY HEATMAP BUILDER ─────────────────────────────────────────────────────

def create_monthly_heatmap(csv_file: str,
                           variant: str,
                           parameter: str,
                           depth: float,
                           tolerance: float,
                           resolution: int,
                           lat_range: tuple,
                           lon_range: tuple,
                           prepared_land_geoms):
    """
    Build and save a monthly heatmap of ARGO – SINMOD differences for a given variant, depth, and parameter.

    Steps:
      1) Parse year & month from the CSV filename (e.g., "standard_201902.csv" → 2019, 02).
      2) Load and filter the DataFrame:
           - Drop rows with NaN or zero values in the relevant columns (temperature or salinity).
           - Keep only casts whose ARGO depth lies within [depth - tolerance, depth + tolerance].
           - Keep only casts within the specified lat/lon bounds.
      3) Compute the pointwise difference (ARGO_avg - SINMOD_avg) at each unique (lat, lon).
      4) Interpolate those pointwise differences onto a regular grid via `interpolate_for_heatmap`.
      5) Load SINMOD bathymetry from the corresponding SINMOD NetCDF file to compute isobaths (if available).
      6) Call `plot_heatmap` to draw and save the final map.

    Parameters
    ----------
    csv_file : str
        Path to the ARGO‐SINMOD monthly CSV (e.g., "standard_201902.csv").
    variant : str
        One of the SINMOD variants (keys of HEATMAP_INPUT_DIRS and HEATMAP_SINMOD_FILE).
    parameter : str
        Either "Temperature" or "Salinity" (determines which columns to use).
    depth : float
        Target depth (m) at which to compute differences.
    tolerance : float
        Depth tolerance (m) around the target depth.
    resolution : int
        Number of grid points along each axis for interpolation.
    lat_range : tuple (lat_min, lat_max)
        Latitude bounds for the map.
    lon_range : tuple (lon_min, lon_max)
        Longitude bounds for the map.
    prepared_land_geoms : list of Shapely geometry
        Pre‐prepared land polygons for masking.

    Raises
    ------
    RuntimeError
        If the CSV filename does not follow the expected pattern to extract year/month.
    """
    # 1) Parse year and month from the CSV stem (e.g. "assimilated_201902" → yr=2019, mo=02)
    stem = Path(csv_file).stem
    try:
        _, ym = stem.rsplit("_", 1)
        yr = int(ym[:4])
        mo = int(ym[4:6])
    except Exception:
        raise RuntimeError(f"Cannot parse year/month from '{stem}'")
    month_name = calendar.month_name[mo]

    print(f"{variant:>10} | {parameter:>11} | {depth:6.1f}m | {yr}-{mo:02d} → {month_name}")

    # 2) Load & filter the DataFrame
    df = load_comparison_data(csv_file)
    if df.empty:
        print("   no data; skipping")
        return

    # Drop rows with NaN in required columns
    df = df.dropna(subset=[
        'ARGO Depth',
        'Temp ARGO (°C)', 'Temp SINMOD (°C)',
        'Salinity ARGO (PSU)', 'Salinity SINMOD (PSU)'
    ])

    # Further drop rows where either ARGO or SINMOD value is zero (invalid):
    if parameter == 'Temperature':
        df = df[(df['Temp ARGO (°C)'] != 0) & (df['Temp SINMOD (°C)'] != 0)]
    else:
        df = df[(df['Salinity ARGO (PSU)'] != 0) & (df['Salinity SINMOD (PSU)'] != 0)]

    # Filter by depth tolerance
    dmask = df['ARGO Depth'].between(depth - tolerance, depth + tolerance)
    df = df[dmask]

    # Filter by geographic bounds if provided
    if lat_range:
        df = df[df['ARGO Lat'].between(*lat_range)]
    if lon_range:
        df = df[df['ARGO Lon'].between(*lon_range)]

    if df.empty:
        print("   no records within depth/bounds; skipping")
        return

    # 3) Group by (lat, lon) and compute mean difference
    if parameter == 'Temperature':
        col_a = 'Temp ARGO (°C)'
        col_s = 'Temp SINMOD (°C)'
    else:
        col_a = 'Salinity ARGO (PSU)'
        col_s = 'Salinity SINMOD (PSU)'

    grp = (
        df.groupby(['ARGO Lat', 'ARGO Lon'], as_index=False)
          .agg(argo_avg=(col_a, 'mean'), sinmod_avg=(col_s, 'mean'))
    )
    vals = grp['argo_avg'] - grp['sinmod_avg']
    lats = grp['ARGO Lat'].values
    lons = grp['ARGO Lon'].values

    # 4) Interpolate to a regular grid and mask land
    XI, YI, VI = interpolate_for_heatmap(
        lats, lons, vals,
        resolution=resolution,
        lat_range=lat_range,
        lon_range=lon_range,
        prepared_land_geoms=prepared_land_geoms
    )

    # 5) Attempt to load bathymetry (contour depths) from SINMOD file
    sinmod_fp = HEATMAP_SINMOD_FILE[variant]
    if sinmod_fp.exists():
        ds = xr.open_dataset(sinmod_fp, engine="h5netcdf")
        proj = get_sinmod_projection(ds)
        xc = ds['xc'].values
        yc = ds['yc'].values
        # SINMOD’s vertical coordinate may be named 'depth' or 'zc'
        zb = ds['depth'].values if 'depth' in ds else ds['zc'].values

        # Build a geographic (lon_g, lat_g) mesh for SINMOD depths
        Xg, Yg = np.meshgrid(xc, yc)
        transformer = Transformer.from_proj(proj, "epsg:4326", always_xy=True)
        lon_g, lat_g = transformer.transform(Xg.flatten(), Yg.flatten())
        CD = griddata((lon_g, lat_g), zb.flatten(), (XI, YI), method='linear')
        ds.close()
    else:
        print(f"   ⚠️  SINMOD file not found: {sinmod_fp}; skipping bathy contours")
        CD = None

    # 6) Prepare output folder & filename
    outdir = HEATMAP_BASE_DIR / variant / parameter.lower() / f"{int(depth)}m"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{yr}{mo:02d}_{parameter.lower()}_{int(depth)}m.png"
    outfp = outdir / fname

    # 7) Plot and save
    plot_heatmap(
        XI, YI, VI,
        depth, parameter, tolerance,
        lats, lons,    # scatter coordinates
        CD,            # bathymetry contours (or None)
        outfp
    )


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    """
    Driver function that loops over each SINMOD variant and each monthly CSV file
    to produce heatmaps. Default parameters are set here, but can be modified or
    exposed via an argument parser if desired.
    """
    # Default parameters (depth in m, tolerance in m, grid resolution, lat/lon bounds)
    parameter = "Temperature"   # or "Salinity"
    depth     = 50.0            # depth in meters at which to compare
    tolerance = 50.0             # depth tolerance in meters
    resolution= 200             # number of grid cells along each axis (200×200)
    lat_range = (45, 80)        # latitude bounds for interpolation and plotting
    lon_range = (-30, 30)       # longitude bounds for interpolation and plotting

    # Pre‐prepare land polygons (Shapely) once, to speed up masking
    land_geoms = list(cfeature.LAND.geometries())
    prepared_land_geoms = [prep(g) for g in land_geoms]

    # Loop over each SINMOD variant’s input directory of CSVs
    for variant, indir in HEATMAP_INPUT_DIRS.items():
        pattern = str(indir / f"{variant}_*.csv")
        files = sorted(glob.glob(pattern))
        print(f"\n>> {variant.upper()}: found {len(files)} CSV files")

        for csv_file in files:
            create_monthly_heatmap(
                csv_file,
                variant,
                parameter, depth, tolerance,
                resolution,
                lat_range, lon_range,
                prepared_land_geoms
            )

if __name__ == "__main__":
    main()
