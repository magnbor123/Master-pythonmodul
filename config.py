from pathlib import Path

"""
config.py

Central configuration file for ARGO vs. SINMOD comparisons and heatmap/profile generation.

Defines:
  - INPUT_BASE:   Base folder containing all raw datasets (ARGO, SINMOD, Copernicus, etc.).
  - ARGO_DIR:     Directory for ARGO NetCDF profile files.
  - SINMOD_DIR:   Directory for standard SINMOD NetCDF files.
  - NUDGE_ASSIM_DIR: Directory for nudged/assimilated SINMOD outputs.
  - COPERNICUS_DIR:  Directory for Copernicus SST NetCDF files.

  - SINMOD_FILES: Mapping of file identifiers to actual NetCDF file paths.
  - get_sinmod_file(): Function to choose the correct SINMOD file based on variant and year.

  - OUTPUT_BASE:  Base directory for all derived outputs (CSV comparisons, heatmaps, profiles).
  - RESULTS_BASE: Subdirectory under OUTPUT_BASE for ARGO–SINMOD CSV outputs.

  - OUTPUT_DIRS:  Dictionary mapping each SINMOD variant ("standard", "nudged", "assimilated")
                  → its dedicated folder under RESULTS_BASE for monthly comparison CSVs.

  - YEARS, MONTHS: Lists specifying which years and months to process.
  - TIME_TOL_HOURS: Time‐matching tolerance (hours) when pairing ARGO casts to SST/SINMOD snapshots.
  - MAX_DIST_DEG:   Maximum allowed horizontal degree‐difference when matching grid cells.

  - HEATMAP_INPUT_DIRS: Identical to OUTPUT_DIRS; used by heatmap scripts to locate monthly CSVs.
  - HEATMAP_BASE_DIR:   Base directory under OUTPUT_BASE/results/heatmaps to save heatmap PNGs.
  - HEATMAP_SINMOD_FILE: Mapping of each variant → the SINMOD file to use for bathymetry contours.

  - VERT_DEPTH_MIN:   Minimum ARGO cast depth (m) to include in vertical‐profile plots.
  - VERT_COAST_KM:    Maximum distance (km) from coast to consider a cast “coastal”.
  - VERT_PROFILE_BASE: Directory under OUTPUT_BASE/results/vertical_profiles to save profile plots.
  - MAX_PROFILES_PER_MONTH: Maximum number of offshore profiles to plot per month.

  - SST_DIR:          Alias to COPERNICUS_DIR; used by profile scripts to load SST values.
"""


# ─── INPUT DATA ────────────────────────────────────────────────────────────────
INPUT_BASE       = Path("D:/Magnus Børslid")
"""
Base directory containing:
  - Argo_DATA/
  - Sinmod_DATA/
  - nudge_assim/
  - Copernicus/
"""

ARGO_DIR         = INPUT_BASE / "Argo_DATA"
"""Directory with ARGO NetCDF profile files."""

SINMOD_DIR       = INPUT_BASE / "Sinmod_DATA"
"""Directory with standard (uninfluenced) SINMOD NetCDF files."""

NUDGE_ASSIM_DIR  = INPUT_BASE / "nudge_assim"
"""Directory with nudged and assimilated SINMOD NetCDF files."""

COPERNICUS_DIR   = INPUT_BASE / "Copernicus"
"""Directory containing Copernicus SST NetCDF files."""


# ─── SINMOD FILES ───────────────────────────────────────────────────────────────
SINMOD_FILES = {
    "standard_2019":  SINMOD_DIR      / "nor4km_PhysStates2019.nc",
    "standard_22_23": SINMOD_DIR      / "PhysStates_2022_2023_new.nc",
    "nudged":         NUDGE_ASSIM_DIR / "PhysStates_nudged_2019.nc",
    "assimilated":    NUDGE_ASSIM_DIR / "PhysStates_assim_2019.nc",
}
"""
Mapping of variant keys to their respective SINMOD NetCDF file paths:
  - "standard_2019": Standard (no assimilation) SINMOD for year 2019.
  - "standard_22_23": Standard SINMOD covering 2022–2023.
  - "nudged": Nudged SINMOD output for 2019.
  - "assimilated": Assimilated SINMOD output for 2019.
"""


def get_sinmod_file(variant: str, year: int) -> Path:
    """
    Select the correct SINMOD NetCDF file for a given variant and year.

    Parameters
    ----------
    variant : str
        One of "standard", "nudged", or "assimilated".
    year : int
        Four‐digit year (e.g. 2019, 2022, 2023).

    Returns
    -------
    Path
        Path to the corresponding SINMOD NetCDF file.

    Raises
    ------
    ValueError
        If requesting the "standard" variant for a year not covered (neither 2019 nor 2022/2023).
    KeyError
        If given an unknown variant string.
    """
    if variant == "standard":
        if year == 2019:
            return SINMOD_FILES["standard_2019"]
        elif year in (2022, 2023):
            return SINMOD_FILES["standard_22_23"]
        else:
            raise ValueError(f"No standard SINMOD file for year {year}")
    elif variant in ("nudged", "assimilated"):
        # Both nudged and assimilated use the 2019 file (fallback if year != 2019 is not typical)
        return SINMOD_FILES[variant]
    else:
        raise KeyError(f"Unknown variant '{variant}'")


# ─── OUTPUT DIRECTORIES ─────────────────────────────────────────────────────────
OUTPUT_BASE  = Path("D:/pythonmodul")
"""
Root folder for all output products (CSVs, heatmaps, profiles). 
Example structure:
  D:/pythonmodul/
    results_argo_sinmod/
    results/heatmaps/
    results/vertical_profiles/
"""

RESULTS_BASE = OUTPUT_BASE / "results_argo_sinmod"
"""Parent folder under which ARGO–SINMOD comparison CSVs (per variant) will be stored."""

# Create one subdirectory per SINMOD variant for CSV outputs
OUTPUT_DIRS = {
    variant: (RESULTS_BASE / variant)
    for variant in ("standard", "nudged", "assimilated")
}
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
"""
OUTPUT_DIRS: dict
  - Keys: "standard", "nudged", "assimilated"
  - Values: Path objects under RESULTS_BASE where monthly comparison CSVs should be saved.
"""


# ─── RUN CONTROL ───────────────────────────────────────────────────────────────
YEARS          = [2019, 2022, 2023]
"""
List of years to process. Script loops over these to match ARGO, SINMOD, Copernicus data.
"""

MONTHS         = list(range(1, 13))
"""
Months (1–12) to process in each year.
"""

TIME_TOL_HOURS = 12
"""Time tolerance (hours) when matching ARGO casts to SST or SINMOD snapshots."""

MAX_DIST_DEG   = 0.05
"""Maximum degree difference (°) when matching points between model grids (used for horizontal matching)."""


# ─── HEATMAP SETTINGS ───────────────────────────────────────────────────────────
HEATMAP_INPUT_DIRS = OUTPUT_DIRS
"""
Dictionary of input folders for heatmap scripts, identical to OUTPUT_DIRS.
Heatmap generation scripts read from these CSV directories.
"""

HEATMAP_BASE_DIR   = OUTPUT_BASE / "results" / "heatmaps"
HEATMAP_BASE_DIR.mkdir(parents=True, exist_ok=True)
"""
Base directory under OUTPUT_BASE/results/ where all heatmap PNGs will be saved.
Subfolders will be created per variant and parameter (e.g., "sst_standard", "sst_nudged", etc.).
"""

# Which SINMOD file to use for bathymetry contours in heatmap scripts (choose first year in YEARS list)
HEATMAP_SINMOD_FILE = {
    variant: get_sinmod_file(variant, YEARS[0])
    for variant in OUTPUT_DIRS
}
"""
HEATMAP_SINMOD_FILE: dict
  - Keys: SINMOD variant names
  - Values: Path to the SINMOD NetCDF file used for retrieving bathymetry/depth grids
    (only the first year in YEARS is used, since bathymetry does not change year‐over‐year).
"""


# ─── VERTICAL PROFILES SETTINGS ────────────────────────────────────────────────
VERT_DEPTH_MIN        = 50.0
"""
Minimum ARGO cast depth (m) to include when making vertical profile plots.
Only casts with depth ≥ this value will be considered (e.g., 50 m).
"""

VERT_COAST_KM         = 10.0
"""
Maximum distance from coastline (km) for “coastal” casts in vertical profile scripts.
Casts farther than this will be treated as “offshore” in separate scripts.
"""

VERT_PROFILE_BASE     = OUTPUT_BASE / "results" / "vertical_profiles"
VERT_PROFILE_BASE.mkdir(parents=True, exist_ok=True)
"""
Base directory under which vertical profile PNGs will be saved.
Structure typically:
  results/vertical_profiles/
    depth_{DEPTH}m/
      coastal/ or offshore/
        YYYYMM/
          profile_YYYYMM_DD_LAT_LON.png
"""

# How many offshore profiles to plot per month (randomly sampled if more are available)
MAX_PROFILES_PER_MONTH = 15


# ─── SST (Copernicus) ──────────────────────────────────────────────────────────
SST_DIR = COPERNICUS_DIR
"""
Alias for the Copernicus SST directory, used by profile scripts (e.g. load_sst_value).
"""
