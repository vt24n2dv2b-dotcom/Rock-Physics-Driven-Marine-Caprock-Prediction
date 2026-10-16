# -*- coding: utf-8 -*-
"""
Draw 12 independent well-log figures directly from the updated TXT file.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==========================================================
# Global font/style settings
# ==========================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# ==========================================================
# User settings
# ==========================================================
TXT_FILE = "Calculated_parameters.txt"
OUTPUT_DIR = "Figure"

# You can modify all x-limits here later.
PI_XLIM = (300, 700)
POR_XLIM = (0, 15)
VQUAR_XLIM = (50, 100)
VCLC_XLIM = (0, 50)
VCLAY_XLIM = (0, 35)
LITH_XLIM = (0, 10)
VP_XLIM = (3500, 5500)
VS_XLIM = (2000, 4000)
RHOB_XLIM = (2300, 2700)
EI1_XLIM = (1e7, 1.4e7)
EI2_XLIM = (1e7, 1.4e7)
EI3_XLIM = (1e7, 1.4e7)

# Lithology color mapping:
# 1 = Sandstone, 2 = Mudstone, 3 = Calcareous sandstone
LITH_COLORS = ["yellow", "black", "blue"]

# ==========================================================
# Locate and read TXT
# ==========================================================
script_dir = Path(__file__).resolve().parent
cwd = Path.cwd()

candidate_paths = [
    script_dir / TXT_FILE,
    cwd / TXT_FILE,
]

txt_path = None
for p in candidate_paths:
    if p.exists():
        txt_path = p
        break

if txt_path is None:
    raise FileNotFoundError(
        f"TXT file not found. Place '{TXT_FILE}' in the script directory or current working directory."
    )

data = pd.read_csv(txt_path, sep=r"\s+", header=None, engine="python")
data = data.dropna(axis=1, how="all")

expected_ncols = 15
if data.shape[1] != expected_ncols:
    raise ValueError(
        f"Expected {expected_ncols} columns in TXT, but got {data.shape[1]}.\n"
        "Expected order:\n"
        "Depth, Vp_obs, Vp_cal, Vs_obs, Vs_cal, EI1, EI2, EI3, "
        "VQUA, Vclay, VCLC, PI, por, Lith, RHOB"
    )

columns = [
    "Depth",
    "Vp_obs",
    "Vp_cal",
    "Vs_obs",
    "Vs_cal",
    "EI1",
    "EI2",
    "EI3",
    "VQUA",
    "Vclay",
    "VCLC",
    "PI",
    "por",
    "Lith",
    "RHOB",
]

data.columns = columns
for c in columns:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna().reset_index(drop=True)

# ==========================================================
# Prepare arrays
# ==========================================================
depth1 = data["Depth"].to_numpy()
vp_obs = data["Vp_obs"].to_numpy()
vp_cal = data["Vp_cal"].to_numpy()
vs_obs = data["Vs_obs"].to_numpy()
vs_cal = data["Vs_cal"].to_numpy()
ei1 = data["EI1"].to_numpy()
ei2 = data["EI2"].to_numpy()
ei3 = data["EI3"].to_numpy()
vquar = data["VQUA"].to_numpy() * 100.0
vclay = data["Vclay"].to_numpy() * 100.0
vclc = data["VCLC"].to_numpy() * 100.0
pi_curve = data["PI"].to_numpy()
por_curve = data["por"].to_numpy() * 100.0
lith = data["Lith"].to_numpy().astype(int)
rhob = data["RHOB"].to_numpy()

outdir = script_dir / OUTPUT_DIR
outdir.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ---------a----------- PI
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(pi_curve, depth1, color='gray', linewidth=2)
ax.set_xlim(PI_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('PI (kg/m$^3$)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'PI.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------b----------- Porosity
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(por_curve, depth1, color='gray', linewidth=2)
ax.set_xlim(POR_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Porosity (%)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'por.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------c----------- Vquar
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(vquar, depth1, color='gray', linewidth=2)
ax.set_xlim(VQUAR_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vquar (%)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'Vquar.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------d----------- Vcalcite
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(vclc, depth1, color='gray', linewidth=2)
ax.set_xlim(VCLC_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vcalcite (%)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'Vclc.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------e----------- Vclay
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(vclay, depth1, color='gray', linewidth=2)
ax.set_xlim(VCLAY_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vclay (%)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'Vclay.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------f----------- Lithology
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
lith_image = np.repeat(lith[:, None], 100, axis=1)
cmap = ListedColormap(LITH_COLORS)
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
ax.imshow(
    lith_image,
    aspect='auto',
    cmap=cmap,
    norm=norm,
    extent=(LITH_XLIM[0], LITH_XLIM[1], np.max(depth1), np.min(depth1)),
    interpolation='nearest'
)
ax.set_xlim(LITH_XLIM)
ax.set_ylim(np.max(depth1), np.min(depth1))
ax.set_ylabel('Depth (m)')
ax.set_xlabel('True lithology (Train)')
ax.tick_params(direction='out', top=False, right=False, width=1.0, labelbottom=False)
plt.savefig(outdir / 'Lith.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()

# ==========================================================
# ---------g----------- Vp
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(vp_obs, depth1, color='gray', linewidth=2, label='Observed')
ax.plot(vp_cal, depth1, '--', color='r', linewidth=2, label='Calculated')
ax.set_xlim(VP_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vp (m/s)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
ax.legend(frameon=False)
plt.savefig(outdir / 'Vp.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()


# ==========================================================
# ---------h----------- Vs
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(vs_obs, depth1, color='gray', linewidth=2, label='Observed')
ax.plot(vs_cal, depth1, '--', color='r', linewidth=2, label='Calculated')
ax.set_xlim(VS_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Vs (m/s)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
ax.legend(frameon=False)
plt.savefig(outdir / 'Vs.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()

# ==========================================================
# ---------i----------- Density
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(rhob, depth1, color='gray', linewidth=2)
ax.set_xlim(RHOB_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Density (kg/m$^3$)')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'Density.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()

# ==========================================================
# ---------j----------- EI1
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(ei1, depth1, color='gray', linewidth=2)
ax.set_xlim(EI1_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('EI1 (kg/m$^2$·s))')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'EI1.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()

# ==========================================================
# ---------k----------- EI2
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(ei2, depth1, color='gray', linewidth=2)
ax.set_xlim(EI2_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('EI2 (kg/m$^2$·s))')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'EI2.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
# #plt.show()

# ==========================================================
# ---------l----------- EI3
# ==========================================================
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 7))
ax.plot(ei3, depth1, color='gray', linewidth=2)
ax.set_xlim(EI3_XLIM)
#ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_xlabel('EI3 (kg/m$^2$·s))')
ax.set_ylim(depth1.max(), depth1.min())
ax.tick_params(direction='out', top=False, right=False, width=1.0)
plt.savefig(outdir / 'EI3.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
#plt.show()

print(f'TXT file used: {txt_path}')
print(f'Output directory: {outdir}')
print('Finished drawing 12 independent well-log figures from TXT only.')
