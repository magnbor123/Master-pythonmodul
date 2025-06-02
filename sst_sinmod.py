#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Proj
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config

# ------------------------------------------------------------------
# Universal dataset opener (tries multiple backends + UNC for Windows)
# ------------------------------------------------------------------
def open_dataset(path: Path, engines=None) -> xr.Dataset:
    """
    Open a NetCDF dataset with xarray, trying multiple engines.
    On Windows, adds the \\?\ prefix to support long/unicode paths.
    """
    p = str(path)
    if os.name == 'nt':
        # Make absolute and prefix \\?\
        p = os.path.abspath(p)
        if not p.startswith('\\\\?\\'):
            p = '\\\\?\\' + p
    last_err = None
    for eng in (engines or ['h5netcdf', 'netcdf4', 'scipy']):
        try:
            ds = xr.open_dataset(p, engine=eng)
            if eng == 'scipy':
                ds = ds.load()   # force load into memory
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to open {path!r} with engines {engines}: {last_err}")

# ------------------------------------------------------------------
# SINMOD & Copernicus helpers
# ------------------------------------------------------------------
def process_sinmod(ds, year, month):
    """
    Convert the SINMOD time array into a DatetimeIndex and
    filter for the specified year and month.
    Returns: sinmod_x, sinmod_y, depths, sinmod_times_filtered
    """
    sinmod_x = ds['xc'].values
    sinmod_y = ds['yc'].values
    depths   = ds['zc'].values
    raw      = ds['time'].values  # shape (n,6)
    dt = pd.to_datetime({
        'year':   raw[:,0],
        'month':  raw[:,1],
        'day':    raw[:,2],
        'hour':   raw[:,3],
        'minute': raw[:,4],
        'second': raw[:,5],
    })
    idx = pd.DatetimeIndex(dt)
    mask = (idx.year == year) & (idx.month == month)
    return sinmod_x, sinmod_y, depths, idx[mask]

def get_sinmod_projection(ds) -> Proj:
    """
    Extract SINMOD grid projection parameters from either a 'grid_mapping' variable
    or from global attributes. Returns a pyproj.Proj object.
    """
    gm = ds.get('grid_mapping', None)
    attrs = (gm.attrs if gm is not None else ds.attrs)
    return Proj(
        proj='stere',
        lat_0=attrs['latitude_of_projection_origin'],
        lon_0=attrs['straight_vertical_longitude_from_pole'],
        lat_ts=attrs['standard_parallel'],
        x_0=attrs['false_easting'],
        y_0=attrs['false_northing'],
        a=attrs['semi_major_axis'],
        b=attrs['semi_minor_axis']
    )

def load_copernicus(path: Path):
    """
    Load a Copernicus SST file, convert SST from Kelvin to °C,
    and return (sst, lats, lons, times).
    """
    ds    = open_dataset(path)
    sst   = ds['analysed_sst'].values - 273.15
    lats  = ds['latitude'].values
    lons  = ds['longitude'].values
    times = pd.to_datetime(ds['time'].values)
    ds.close()
    return sst, lats, lons, times

def match_temporal_data(sst_cop, times_cop, temp_sin, times_sin, tol_hours='6h'):
    """
    For each time in times_cop, find the closest time in times_sin
    within tol_hours (e.g. '6h'). Return matched arrays and matched times.
    """
    matched_sst, matched_sin, matched_times = [], [], []
    tol = pd.Timedelta(tol_hours)
    for t0 in times_cop:
        diffs = abs(times_sin - t0)
        if diffs.min() <= tol:
            i_sin = diffs.argmin()
            i_cop = np.where(times_cop == t0)[0][0]
            matched_sst.append(sst_cop[i_cop])
            matched_sin.append(temp_sin[i_sin])
            matched_times.append(t0)
    return np.array(matched_sst), np.array(matched_sin), matched_times

def plot_difference_heatmap(sst_cop, lats_cop, lons_cop, temp_sin, proj, month_str, out_path: Path):
    """
    Reproject SINMOD temperature onto the Copernicus grid, compute (Copernicus - SINMOD),
    and plot the difference with pcolormesh clipped to [-4, +4] °C on a North Polar Stereographic map.
    """
    # Build full Copernicus lat/lon grid
    Lon, Lat = np.meshgrid(lons_cop, lats_cop)
    cop_pts  = np.column_stack((Lat.ravel(), Lon.ravel()))

    # Convert SINMOD grid (x,y) to (lon,lat)
    global sinmod_x, sinmod_y
    X, Y      = np.meshgrid(sinmod_x, sinmod_y)
    lon_s, lat_s = proj(X, Y, inverse=True)
    sin_pts   = np.column_stack((lat_s.ravel(), lon_s.ravel()))

    # Interpolate SINMOD temperature onto Copernicus grid
    sin_i    = griddata(sin_pts, temp_sin.ravel(), cop_pts, method='linear')
    sin_grid = sin_i.reshape(Lon.shape)

    # Compute difference
    diff = sst_cop - sin_grid

    # Plot with pcolormesh
    fig = plt.figure(figsize=(12,8))
    ax  = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([lons_cop.min(), lons_cop.max(),
                   lats_cop.min(), lats_cop.max()],
                  crs=ccrs.PlateCarree())

    pcm = ax.pcolormesh(
        Lon, Lat, diff,
        cmap='RdBu_r', vmin=-4, vmax=4,
        transform=ccrs.PlateCarree()
    )
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    cb = plt.colorbar(pcm, ax=ax, orientation='vertical', label='Temp Difference (°C)')
    plt.title(f"Temp Diff (Copernicus – SINMOD) for {month_str}")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    cop_files = sorted(
        p for p in config.COPERNICUS_DIR.glob("*.nc")
        if not p.name.startswith("._")
    )

    for variant in config.OUTPUT_DIRS:
        out_dir_var = config.HEATMAP_BASE_DIR / f"sst_{variant}"
        out_dir_var.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Variant: {variant} → Output → {out_dir_var}")

        for cop in cop_files:
            sst, lats, lons, times = load_copernicus(cop)
            years_in = pd.DatetimeIndex(times).year.unique()

            for year in config.YEARS:
                if year not in years_in:
                    continue
                for month in config.MONTHS:
                    mask = (
                        (times.year == year) &
                        (times.month == month) &
                        (times.day == 1)
                    )
                    if not mask.any():
                        continue

                    # Only take the first time on that day
                    sst_day = sst[mask, :, :][0]
                    t0      = pd.DatetimeIndex(times[mask])[0]
                    month_str = f"{year}-{month:02d}"

                    # Choose the SINMOD file according to variant & year
                    sin_file = config.get_sinmod_file(variant, year)

                    # -------------------------------
                    # BEGIN: Filter SINMOD to the target month exactly
                    # -------------------------------
                    # Step A: open SINMOD and get full timestamp array
                    ds = open_dataset(sin_file)
                    raw = ds['time'].values   # shape = (N_total, 6)
                    full_dt = pd.to_datetime({
                        'year':   raw[:,0],
                        'month':  raw[:,1],
                        'day':    raw[:,2],
                        'hour':   raw[:,3],
                        'minute': raw[:,4],
                        'second': raw[:,5],
                    })
                    idx_full = pd.DatetimeIndex(full_dt)

                    # Step B: get (sx, sy, depths, sin_times) for this month
                    sx, sy, depths, sin_times = process_sinmod(ds, year, month)

                    # Step C: find which indices in idx_full correspond to sin_times
                    filtered_indices = [ np.where(idx_full == t)[0][0] for t in sin_times ]

                    # Pick the shallowest (surface) depth
                    shallowest_idx = np.argmin(np.abs(depths))
                    print(f"Available SINMOD depths (m): {depths}")
                    print(f"Using shallowest depth = {depths[shallowest_idx]:.2f} m  (index {shallowest_idx})")

                    # Step D: extract the entire file's surface temps, then subselect the month
                    all_tmp = ds['temperature'][:, shallowest_idx, :, :].values  # shape = (N_total, Ny, Nx)
                    temp_sinmod = all_tmp[filtered_indices, :, :]                # shape = (N_month, Ny, Nx)
                    ds.close()
                    # -------------------------------
                    # END: Filter SINMOD to the target month
                    # -------------------------------

                    # Step E: Now match the single Copernicus time against the month‐filtered SINMOD array
                    sst_m, sin_m, matched = match_temporal_data(
                        np.array([sst_day]),
                        pd.DatetimeIndex([t0]),
                        temp_sinmod,
                        sin_times,
                        tol_hours=f"{config.TIME_TOL_HOURS}h"
                    )
                    if not matched:
                        print(f"No temporal match for {year}-{month:02d}-01 at {t0}; skipping.")
                        continue

                    # Step F: load projection for interpolation
                    ds = open_dataset(sin_file)
                    proj = get_sinmod_projection(ds)
                    ds.close()

                    # Expose sinmod_x, sinmod_y for the plotting function
                    globals()['sinmod_x'] = sx
                    globals()['sinmod_y'] = sy

                    # Step G: Plot and save
                    out_file = out_dir_var / f"{variant}_tempdiff_{month_str}.png"
                    plot_difference_heatmap(
                        sst_m[0], lats, lons, sin_m[0],
                        proj, month_str, out_file
                    )
                    print(f"  ✓ {variant} {month_str} → {out_file.name}")
