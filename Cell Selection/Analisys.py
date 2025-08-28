import numpy as np
import pandas as pd
import pickle
import torch
import os
import math
from CellSelection import Cell, Config
import matplotlib.pyplot as plt

Locations = {
    "Cell" : 0,
    "Pack_Voltage" : 1,
    "SMod" : 2,
    "S" : 3,
    "Pack_Current" : 4,
    "P" : 5,
    "MCnt" : 6,
    "Pack_Power_kW" : 7,
    "Pack_Energy_kWh" : 8,
    "Total_Pack_Weight_kg" : 9,
    "Cell_Weight_kg" : 10,
    "Total_Wall_Weight_kg" : 11,
    "Grav_Energy_Density_Wh_kg" : 12,
    "Grav_Power_Density_W_kg" : 13,
    "Pack_Length_mm" : 14,
    "Pack_Width_mm" : 15,
    "Pack_Height_mm" : 16,
    "Total_Wall_Volume_mm" : 17

}



configs = pd.read_csv("configs.csv")

EArray = configs.iloc[:, Locations["Pack_Energy_kWh"]].to_numpy()
WArray = configs.iloc[:, Locations["Total_Pack_Weight_kg"]].to_numpy()
PArray = configs.iloc[:, Locations["Pack_Power_kW"]].to_numpy()
print("EArray:", EArray)
print("WArray:", WArray)
print("PArray:", PArray)
# Map cell names to unique numbers for coloring
cell_names = configs.iloc[:, Locations["Cell"]].to_numpy()
unique_cells, cell_indices = np.unique(cell_names, return_inverse=True)
# cell_indices now maps each config to a unique cell number


# Prepare MCnt array
MCntArray = configs.iloc[:, Locations["MCnt"]].to_numpy()

# Create 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Top left: Power vs Weight, color by Energy
from matplotlib.colors import LogNorm
from matplotlib import ticker
# Nonlinear scaling function centered at 80kW (Y) and 5kWh (color)
def nonlinear_axis_transform(x, center, scale=1.0):
    return np.sign(x - center) * np.log1p(np.abs(x - center) * scale) + center
# Transform Y (Power) axis for high resolution near 80kW
PArray_nonlinear = nonlinear_axis_transform(PArray, center=80, scale=0.15)
# Transform X (Energy) axis for high resolution near 5kWh
EArray_nonlinear = nonlinear_axis_transform(EArray, center=5, scale=0.15)
sc1 = axs[0, 0].scatter(EArray_nonlinear, PArray_nonlinear, c=WArray, cmap='viridis', alpha=0.6)
fig.colorbar(sc1, ax=axs[0, 0], label='Total Pack Weight (kg)')
axs[0, 0].set_xlabel('Pack Energy (kWh) [nonlinear scale]')
axs[0, 0].set_ylabel('Pack Power (kW) [nonlinear scale]')
axs[0, 0].set_title('Pack Power vs Pack Energy (Color: Pack Weight, Nonlinear Axes)')
# Draw gridlines at constant intervals in the original (linear) data space, transformed to nonlinear axis
energy_grid = np.arange(EArray.min(), EArray.max(), 1)
power_grid = np.arange(PArray.min(), PArray.max(), 10)
for x in energy_grid:
    axs[0, 0].axvline(nonlinear_axis_transform(x, center=5, scale=0.15), color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
for y in power_grid:
    axs[0, 0].axhline(nonlinear_axis_transform(y, center=80, scale=0.15), color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
axs[0, 0].grid(False)
# Set custom ticks for nonlinear axes
energy_ticks = [EArray.min(), 5, EArray.max()]
power_ticks = [PArray.min(), 80, PArray.max()]
axs[0, 0].set_xticks(nonlinear_axis_transform(np.array(energy_ticks), center=5, scale=0.15))
axs[0, 0].set_xticklabels([f'{et:.1f}' for et in energy_ticks])
axs[0, 0].set_yticks(nonlinear_axis_transform(np.array(power_ticks), center=80, scale=0.15))
axs[0, 0].set_yticklabels([f'{pt:.1f}' for pt in power_ticks])

# Top right: Energy vs Weight, color by Power
# Nonlinear X (Pack Weight) axis centered at 30kg, Nonlinear Y (Pack Energy) axis centered at 5, Nonlinear color scale centered at 80kW
WArray_nonlinear_topright = nonlinear_axis_transform(WArray, center=30, scale=2)
EArray_nonlinear_topright = nonlinear_axis_transform(EArray, center=5, scale=2)
PArray_nonlinear_color = nonlinear_axis_transform(PArray, center=80, scale=2)
# Calculate alpha values for each point: fully opaque at 80kW, 90% transparent at furthest from 80kW
center_power = 80
max_transparency = 0.9
min_transparency = 0.0
furthest = np.max(np.abs(PArray - center_power))
# Calculate per-point opacity (make bad values much more transparent)
norm_dist = (np.abs(PArray - center_power) / furthest) ** 8  # Use higher power for sharper dropoff
# Clamp values to [0, 1]
norm_dist = np.clip(norm_dist, 0, 1)
opacity_array = 1.0 - (min_transparency + (max_transparency - min_transparency) * norm_dist)
# Map color values to RGBA and set alpha channel
from matplotlib import cm, colors
norm = colors.Normalize(vmin=PArray_nonlinear_color.min(), vmax=PArray_nonlinear_color.max())
cmap = cm.get_cmap('twilight')
# Create scatter with colormap and norm for colorbar
sc2 = axs[0, 1].scatter(WArray_nonlinear_topright, EArray_nonlinear_topright, c=PArray_nonlinear_color, cmap=cmap, norm=norm)
# Set per-point alpha after creation
rgba_colors = sc2.get_facecolors()
if len(rgba_colors) == len(opacity_array):
    rgba_colors[:, 3] = opacity_array
    sc2.set_facecolor(rgba_colors)
cb2 = fig.colorbar(sc2, ax=axs[0, 1])
cb2.set_label('Pack Power (kW), nonlinear color scale (original range: ~40-180)')
# Set colorbar ticks to show original kW values
import matplotlib.ticker as mticker
color_ticks = [40, 80, 120, 180]
cb2.set_ticks(nonlinear_axis_transform(np.array(color_ticks), center=80, scale=2))
cb2.set_ticklabels([str(v) for v in color_ticks])
axs[0, 1].set_xlabel('Total Pack Weight (kg) [nonlinear scale]')
axs[0, 1].set_ylabel('Pack Energy (kWh) [nonlinear scale]')
axs[0, 1].set_title('Pack Energy vs Pack Weight (Color: Power, Nonlinear X & Y & Color)')
# Draw gridlines at constant intervals in the original space, then scale them to demonstrate the nonlinear nature
weight_grid_linear = np.linspace(WArray.min(), WArray.max(), num=8)
energy_grid_linear = np.linspace(EArray.min(), EArray.max(), num=8)
for x in weight_grid_linear:
    axs[0, 1].axvline(nonlinear_axis_transform(x, center=30, scale=2), color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
for y in energy_grid_linear:
    axs[0, 1].axhline(nonlinear_axis_transform(y, center=5, scale=2), color='lightgray', linestyle='--', linewidth=0.5, zorder=0)
axs[0, 1].grid(False)
# Set custom ticks for nonlinear axes
weight_ticks = [WArray.min(), 30, WArray.max()]
energy_ticks = [EArray.min(), 5, EArray.max()]
axs[0, 1].set_xticks(nonlinear_axis_transform(np.array(weight_ticks), center=30, scale=2))
axs[0, 1].set_xticklabels([f'{wt:.1f}' for wt in weight_ticks])
axs[0, 1].set_yticks(nonlinear_axis_transform(np.array(energy_ticks), center=5, scale=2))
axs[0, 1].set_yticklabels([f'{et:.1f}' for et in energy_ticks])

# Bottom left: Power vs Energy, color by Weight
sc3 = axs[1, 0].scatter(EArray, PArray, c=WArray, cmap='cividis', alpha=0.6, norm=LogNorm(vmin=max(WArray.min(), 1e-6), vmax=WArray.max()))
fig.colorbar(sc3, ax=axs[1, 0], label='Total Pack Weight (kg)', norm=LogNorm(vmin=max(WArray.min(), 1e-6), vmax=WArray.max()))
axs[1, 0].set_xlabel('Pack Energy (kWh)')
axs[1, 0].set_ylabel('Pack Power (kW)')
axs[1, 0].set_title('Pack Power vs Pack Weight (Color: Pack Energy, Nonlinear Axes)')
axs[1, 0].grid(True)


# Bottom right: Cell name (as number) vs Pack Energy, color by Pack Power
sc4 = axs[1, 1].scatter(EArray, PArray, c=cell_indices, cmap='viridis', alpha=0.6)
fig.colorbar(sc4, ax=axs[1, 1], label='Cell Index')
axs[1, 1].set_xlabel('Pack Energy (kWh)')
axs[1, 1].set_ylabel('Pack Power (kW)')
axs[1, 1].set_title('Power vs Energy (Color: Cell Index)')
axs[1, 1].grid(False)

# Standalone plot: Power vs Energy, color by Cell Index (smoother color scale)

# Duplicate the full 2x2 multiplot with linear scales and simple colorbars
fig2, axs2 = plt.subplots(2, 2, figsize=(14, 12))
# Top left: Power vs Energy, color by Weight (linear)
sc1_lin = axs2[0, 0].scatter(EArray, PArray, c=WArray, cmap='viridis', alpha=0.6)
fig2.colorbar(sc1_lin, ax=axs2[0, 0], label='Total Pack Weight (kg)')
axs2[0, 0].set_xlabel('Pack Energy (kWh)')
axs2[0, 0].set_ylabel('Pack Power (kW)')
axs2[0, 0].set_title('Pack Power vs Pack Energy (Color: Pack Weight, Linear Axes)')
axs2[0, 0].grid(True)
# Top right: Energy vs Weight, color by Power (linear)
sc2_lin = axs2[0, 1].scatter(WArray, EArray, c=PArray, cmap='twilight', alpha=0.6)
fig2.colorbar(sc2_lin, ax=axs2[0, 1], label='Pack Power (kW)')
axs2[0, 1].set_xlabel('Total Pack Weight (kg)')
axs2[0, 1].set_ylabel('Pack Energy (kWh)')
axs2[0, 1].set_title('Pack Energy vs Pack Weight (Color: Power, Linear Axes)')
axs2[0, 1].grid(True)
# Bottom left: Power vs Energy, color by Weight (linear)
sc3_lin = axs2[1, 0].scatter(EArray, PArray, c=WArray, cmap='cividis', alpha=0.6)
fig2.colorbar(sc3_lin, ax=axs2[1, 0], label='Total Pack Weight (kg)')
axs2[1, 0].set_xlabel('Pack Energy (kWh)')
axs2[1, 0].set_ylabel('Pack Power (kW)')
axs2[1, 0].set_title('Pack Power vs Pack Weight (Color: Pack Energy, Linear Axes)')
axs2[1, 0].grid(True)
# Bottom right: Cell name (as number) vs Pack Energy, color by Pack Power (linear)
sc4_lin = axs2[1, 1].scatter(EArray, PArray, c=cell_indices, cmap='viridis', alpha=0.6)
fig2.colorbar(sc4_lin, ax=axs2[1, 1], label='Cell Index')
axs2[1, 1].set_xlabel('Pack Energy (kWh)')
axs2[1, 1].set_ylabel('Pack Power (kW)')
axs2[1, 1].set_title('Power vs Energy (Color: Cell Index, Linear Axes)')
axs2[1, 1].grid(True)

plt.tight_layout()
plt.show()


