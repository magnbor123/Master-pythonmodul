# Master-pythonmodul

Suite of scripts to compare ARGO profiles, SINMOD outputs, and Copernicus SST.

## Structure
config.py
argo_sinmod_compare.py
argo_sinmod_heatmaps.py
argo_sst_compare.py
argo_sst_heatmaps.py
coastal_vertical_profiles.py
offshore_vertical_profiles.py
sst_sinmod.py

markdown
Kopier
Rediger

## Prerequisites
- Python 3.8+
- `pip install numpy pandas xarray pyproj scipy matplotlib cartopy shapely`  
  (or `conda install -c conda-forge cartopy shapely pyproj`)

## Configuration
Edit `config.py` to set:
- `INPUT_BASE`, `ARGO_DIR`, `SINMOD_DIR`, `COPERNICUS_DIR`
- `YEARS`, `MONTHS`, `TIME_TOL_HOURS`
- Output paths under `OUTPUT_BASE`

## Usage
```bash
cd /d D:/pythonmodul
python argo_sinmod_compare.py      # CSVs: results_argo_sinmod/
python argo_sinmod_heatmaps.py     # PNGs: results/heatmaps/... 
python argo_sst_compare.py         # CSVs: argo_sst_flat/
python argo_sst_heatmaps.py        # PNGs: results/heatmaps/argo_sst/
python coastal_vertical_profiles.py   # plots: results/vertical_profiles/coastal/
python offshore_vertical_profiles.py  # plots: results/vertical_profiles/offshore/
python sst_sinmod.py               # PNGs: results/heatmaps/sst_{variant}/
