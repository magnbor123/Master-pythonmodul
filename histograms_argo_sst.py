#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# --- Configuration ---
# Directory containing flat ARGO–SST CSVs
FLAT_CSV_DIR = Path("D:/Magnus Børslid/clean_results_argo_sst")
# Output directory for histograms and LaTeX tables
OUTPUT_DIR = FLAT_CSV_DIR / "histograms_argo_sst.py"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Season mapping ---
def get_seasons():
    return {
        'Winter': [1, 2, 3],
        'Spring': [4, 5, 6],
        'Summer': [7, 8, 9],
        'Fall':   [10, 11, 12],
    }

# --- Main processing ---
def create_histograms_and_tex(csv_paths, output_dir, lat_range=None, lon_range=None, tol_hours=12):
    """
    Reads flat ARGO–SST CSVs, computes seasonal & annual histograms
    of the "Difference (°C)" column, and writes a summary LaTeX table.
    """
    seasons = get_seasons()
    # load and concatenate
    df_all = []
    for f in csv_paths:
        df = pd.read_csv(f)
        df['Day']   = pd.to_datetime(df['Day'])
        df['Month'] = df['Day'].dt.month
        df['Year']  = df['Day'].dt.year
        df_all.append(df)
    df = pd.concat(df_all, ignore_index=True)
    # optional spatial filter
    if lat_range:
        df = df[(df['ARGO Lat'] >= lat_range[0]) & (df['ARGO Lat'] <= lat_range[1])]
    if lon_range:
        df = df[(df['ARGO Lon'] >= lon_range[0]) & (df['ARGO Lon'] <= lon_range[1])]
    # ensure difference column exists
    df['Diff'] = df['Difference (°C)']

    # prepare LaTeX table
    tex_path = output_dir / "argo_sst_diff_summary.tex"
    with open(tex_path, 'w') as tex:
        tex.write("\\begin{table}[h!]\n\\centering\n")
        tex.write("\\caption{Seasonal and annual summary of ARGO–SST temperature differences}\n")
        tex.write("\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n")
        tex.write("Season & Year & Mean (°C) & Median (°C) & Count \\\\\n\\hline\n")

        # loop seasons
        for season, months in seasons.items():
            season_df = df[df['Month'].isin(months)]
            if season_df.empty:
                continue
            years = sorted(season_df['Year'].unique())
            # create subplot grid for this season
            fig, axes = plt.subplots(1, len(years), figsize=(5*len(years),4), sharey=True)
            if len(years) == 1:
                axes = [axes]
            fig.suptitle(f"ARGO–SST ΔT Histograms: {season}", fontsize=16)

            for ax, year in zip(axes, years):
                ydf = season_df[season_df['Year']==year]
                if ydf.empty:
                    continue
                mean = ydf['Diff'].mean()
                med  = ydf['Diff'].median()
                cnt  = len(ydf)
                tex.write(f"{season} & {year} & {mean:.2f} & {med:.2f} & {cnt} \\\\\n\\hline\n")

                ax.hist(ydf['Diff'], bins=20, range=(-4,4),
                        edgecolor='black', alpha=0.7, density=True)
                ax.set_title(str(year))
                ax.set_xlabel("ΔT (°C)")
                ax.set_xlim(-4,4)
                ax.grid(True, linestyle='--', linewidth=0.5)
                if ax is axes[0]:
                    ax.set_ylabel("Normalized Frequency")

            # save histogram figure for this season
            hist_path = output_dir / f"histograms_{season.lower()}.png"
            plt.tight_layout(rect=[0,0,1,0.92])
            plt.savefig(hist_path, dpi=300)
            plt.close(fig)

        tex.write("\\end{tabular}\n\\end{table}\n")

    print(f"Generated histograms and LaTeX summary at {OUTPUT_DIR}")

if __name__ == "__main__":
    # find all flat CSVs
    csv_files = sorted(FLAT_CSV_DIR.glob("argo_sst_cmp_*.csv"))
    create_histograms_and_tex(csv_files, OUTPUT_DIR,
                              lat_range=(45,75), lon_range=(-30,30))
