import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import datetime
from Analisys import normalize_cell_name

Locations = {
    "Cell": 0,
    "Pack Voltage (V)": 1,
    "SMod": 2,
    "S": 3,
    "Pack Current (A)": 4,
    "P": 5,
    "MCnt": 6,
    "Pack Power (kW)": 7,
    "Pack Energy (kWh)": 8,
    "Total Pack Weight (kg)": 9,
    "Cell Weight (kg)": 10,
    "Total Wall Weight (kg)": 11,
    "Grav Energy Density (Wh/kg)": 12,
    "Grav Power Density (W/kg)": 13,
    "Pack Length (mm)": 14,
    "Pack Width (mm)": 15,
    "Pack Height (mm)": 16,
    "Total Wall Volume (mm³)": 17
}

def plot_optimization_scatter_coords(results, x_range=(3, 7), y_range=(60, 100), show_plot=True, UDT="", output_folder=None):
    """
    Plots optimization results as scatter points:
    x = Optimization Pack Energy, y = Optimization Pack Power
    """
    
    # Split configs by best, second, third
    configs_by_rank = [[], [], []]
    for key, result_data in results.items():
        power, energy = key
        configs = result_data['results']
        for i, config in enumerate(configs):
            configs_by_rank[i].append((power, energy, config))

    # Color maps
    color_list = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#8B00FF', 
                  '#00FFFF', '#FF1493', '#32CD32', '#FFD700', '#8B4513']
    weight_cmap = plt.get_cmap('plasma')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)

    # Calculate global ranges for consistent coloring
    all_configs = [cfg for configs in configs_by_rank for _, _, cfg in configs]
    all_weights = [cfg.iloc[Locations["Total Pack Weight (kg)"]] for cfg in all_configs]
    all_cells = {normalize_cell_name(cfg.iloc[Locations["Cell"]]) for cfg in all_configs}
    
    global_weight_norm = mcolors.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    global_cell_to_color = {cell: color_list[i % len(color_list)] for i, cell in enumerate(sorted(all_cells))}

    # Pre-calculate grid spacing for each column
    grid_spacing = []
    for col in range(3):
        configs = configs_by_rank[col]
        energies = [e for _, e, _ in configs]
        powers = [p for p, _, _ in configs]
        dx = min(np.diff(sorted(set(energies))))
        dy = min(np.diff(sorted(set(powers))))
        grid_spacing.append((dx, dy))

    # Plot each subplot
    rank_titles = ["Best", "Second Best", "Third Best"]
    for col in range(3):
        configs = configs_by_rank[col]
        dx, dy = grid_spacing[col]
        
        for row in range(2):
            ax = axes[row, col]
            
            # Choose coloring scheme based on row
            if row == 0:  # Top row: colored by cell
                colors = [global_cell_to_color[normalize_cell_name(cfg.iloc[Locations["Cell"]])] 
                         for _, _, cfg in configs]
                color_type = "Cell"
            else:  # Bottom row: colored by weight
                colors = [weight_cmap(global_weight_norm(cfg.iloc[Locations["Total Pack Weight (kg)"]])) 
                         for _, _, cfg in configs]
                color_type = "Weight"
            
            # Plot rectangles
            for (power, energy, cfg), color in zip(configs, colors):
                rect = mpatches.Rectangle((energy-dx/2, power-dy/2), dx, dy, color=color, alpha=1.0)
                ax.add_patch(rect)
            
            # Set properties
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_title(f"{rank_titles[col]} Configs (by {color_type})")
            if row == 1:  # Only bottom row gets x-label
                ax.set_xlabel("Optimization Pack Energy (kWh)")
            if col == 0:  # Only leftmost column gets y-label
                ax.set_ylabel("Optimization Pack Power (kW)")

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Add legend for cell types
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cell,
                         markerfacecolor=global_cell_to_color[cell], markersize=10, markeredgecolor='k')
               for cell in sorted(all_cells)]
    fig.legend(handles=handles, title="Cell Type", loc='center left', bbox_to_anchor=(0.86, 0.75))
    
    # Add colorbar for weights
    sm = plt.cm.ScalarMappable(cmap=weight_cmap, norm=global_weight_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Total Pack Weight (kg)')

    # Save plot to specified folder or current directory
    if output_folder:
        filepath = os.path.join(output_folder, f"optimization_plot_grid{UDT}.png")
    else:
        filepath = f"optimization_plot_grid{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Plot saved as {filepath}")
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_comprehensive_analysis(resultsCylDict, resultsAllDict, x_range=(3, 7), y_range=(60, 100), show_plot=True, UDT="", output_folder=None):
    """
    Creates comprehensive analysis plots similar to the attached image:
    - Cylindrical vs All cell comparisons
    - Weight differences
    - Cell type distributions
    """
    
    # Create figure with 3 rows, 5 columns
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    
    # Color settings
    color_list = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#8B00FF', 
                  '#00FFFF', '#FF1493', '#32CD32', '#FFD700', '#8B4513']
    weight_cmap = plt.get_cmap('plasma')
    
    # Process cylindrical results
    cyl_configs_by_rank = [[], [], []]
    for key, result_data in resultsCylDict.items():
        power, energy = key
        configs = result_data['results']
        for i, config in enumerate(configs):
            if config is not None:
                cyl_configs_by_rank[i].append((power, energy, config))
    
    # Process all results
    all_configs_by_rank = [[], [], []]
    for key, result_data in resultsAllDict.items():
        power, energy = key
        configs = result_data['results']
        for i, config in enumerate(configs):
            if config is not None:
                all_configs_by_rank[i].append((power, energy, config))
    
    # Get all unique cells for consistent coloring
    all_cells = set()
    for rank_configs in cyl_configs_by_rank + all_configs_by_rank:
        for _, _, cfg in rank_configs:
            if cfg is not None:
                all_cells.add(normalize_cell_name(cfg.iloc[Locations["Cell"]]))
    
    cell_colors = {cell: color_list[i % len(color_list)] for i, cell in enumerate(sorted(all_cells))}
    
    # Calculate global weight range for consistent coloring
    all_weights = []
    for rank_configs in cyl_configs_by_rank + all_configs_by_rank:
        for _, _, cfg in rank_configs:
            if cfg is not None:
                all_weights.append(cfg.iloc[Locations["Total Pack Weight (kg)"]])
    
    if all_weights:
        global_weight_norm = mcolors.Normalize(vmin=min(all_weights), vmax=max(all_weights))
    else:
        global_weight_norm = mcolors.Normalize(vmin=0, vmax=1)
    
    # Calculate grid spacing for cylindrical data
    if cyl_configs_by_rank[0]:
        cyl_energies = [e for _, e, _ in cyl_configs_by_rank[0]]
        cyl_powers = [p for p, _, _ in cyl_configs_by_rank[0]]
        cyl_dx = min(np.diff(sorted(set(cyl_energies)))) if len(set(cyl_energies)) > 1 else 0.1
        cyl_dy = min(np.diff(sorted(set(cyl_powers)))) if len(set(cyl_powers)) > 1 else 1.0
    else:
        cyl_dx, cyl_dy = 0.1, 1.0
    
    # Calculate grid spacing for all data
    if all_configs_by_rank[0]:
        all_energies = [e for _, e, _ in all_configs_by_rank[0]]
        all_powers = [p for p, _, _ in all_configs_by_rank[0]]
        all_dx = min(np.diff(sorted(set(all_energies)))) if len(set(all_energies)) > 1 else 0.1
        all_dy = min(np.diff(sorted(set(all_powers)))) if len(set(all_powers)) > 1 else 1.0
    else:
        all_dx, all_dy = 0.1, 1.0
    
    # Calculate global weight difference range for colorbar
    all_weight_diffs = []
    for row_idx in range(3):
        cyl_configs = cyl_configs_by_rank[row_idx]
        all_configs = all_configs_by_rank[row_idx]
        
        cyl_dict = {(p, e): cfg.iloc[Locations["Total Pack Weight (kg)"]] 
                   for p, e, cfg in cyl_configs if cfg is not None}
        all_dict = {(p, e): cfg.iloc[Locations["Total Pack Weight (kg)"]] 
                   for p, e, cfg in all_configs if cfg is not None}
        
        for (power, energy), all_weight in all_dict.items():
            if (power, energy) in cyl_dict:
                cyl_weight = cyl_dict[(power, energy)]
                diff = all_weight - cyl_weight
                all_weight_diffs.append(diff)
    
    # Set up global difference normalization starting from 0
    if all_weight_diffs:
        global_diff_min, global_diff_max = min(all_weight_diffs), max(all_weight_diffs)
        global_diff_norm = mcolors.Normalize(vmin=0, vmax=max(abs(global_diff_min), abs(global_diff_max)))
    else:
        global_diff_norm = mcolors.Normalize(vmin=0, vmax=5)
    
    # Plot each rank (Best, Second Best, Third Best)
    rank_titles = ['Best', 'Second Best', 'Third Best']
    col_titles = ['Cylindrical by Cell', 'Cylindrical by Weight', 'Weight Difference', 
                  'All Cells by Weight', 'All Cells by Cell']
    
    for row_idx in range(3):
        # Get configs for this rank
        cyl_configs = cyl_configs_by_rank[row_idx]
        all_configs = all_configs_by_rank[row_idx]
        
        # Plot 1: Cylindrical by Cell
        ax = axes[row_idx, 0]
        for power, energy, cfg in cyl_configs:
            if cfg is not None:
                cell = normalize_cell_name(cfg.iloc[Locations["Cell"]])
                color = cell_colors[cell]
                rect = mpatches.Rectangle((energy-cyl_dx/2, power-cyl_dy/2), cyl_dx, cyl_dy, 
                                        color=color, alpha=1.0)
                ax.add_patch(rect)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=14, fontweight='bold', pad=20)
        
        # Plot 2: Cylindrical by Weight
        ax = axes[row_idx, 1]
        for power, energy, cfg in cyl_configs:
            if cfg is not None:
                weight = cfg.iloc[Locations["Total Pack Weight (kg)"]]
                color = weight_cmap(global_weight_norm(weight))
                rect = mpatches.Rectangle((energy-cyl_dx/2, power-cyl_dy/2), cyl_dx, cyl_dy, 
                                        color=color, alpha=1.0)
                ax.add_patch(rect)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=14, fontweight='bold', pad=20)
        
        # Plot 3: Weight Difference (All - Cylindrical)
        ax = axes[row_idx, 2]
        # Create weight difference visualization
        cyl_dict = {(p, e): cfg.iloc[Locations["Total Pack Weight (kg)"]] 
                   for p, e, cfg in cyl_configs if cfg is not None}
        all_dict = {(p, e): cfg.iloc[Locations["Total Pack Weight (kg)"]] 
                   for p, e, cfg in all_configs if cfg is not None}
        
        # Plot differences using global normalization
        for (power, energy), all_weight in all_dict.items():
            if (power, energy) in cyl_dict:
                cyl_weight = cyl_dict[(power, energy)]
                diff = all_weight - cyl_weight
                
                # Use absolute value of difference with viridis colormap
                color = plt.cm.viridis(global_diff_norm(abs(diff)))
                
                rect = mpatches.Rectangle((energy-all_dx/2, power-all_dy/2), all_dx, all_dy, 
                                        color=color, alpha=1.0)
                ax.add_patch(rect)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=14, fontweight='bold', pad=20)
        
        # Plot 4: All Cells by Weight
        ax = axes[row_idx, 3]
        for power, energy, cfg in all_configs:
            if cfg is not None:
                weight = cfg.iloc[Locations["Total Pack Weight (kg)"]]
                color = weight_cmap(global_weight_norm(weight))
                rect = mpatches.Rectangle((energy-all_dx/2, power-all_dy/2), all_dx, all_dy, 
                                        color=color, alpha=1.0)
                ax.add_patch(rect)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if row_idx == 0:
            ax.set_title(col_titles[3], fontsize=14, fontweight='bold', pad=20)
        
        # Plot 5: All Cells by Cell
        ax = axes[row_idx, 4]
        for power, energy, cfg in all_configs:
            if cfg is not None:
                cell = normalize_cell_name(cfg.iloc[Locations["Cell"]])
                color = cell_colors[cell]
                rect = mpatches.Rectangle((energy-all_dx/2, power-all_dy/2), all_dx, all_dy, 
                                        color=color, alpha=1.0)
                ax.add_patch(rect)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if row_idx == 0:
            ax.set_title(col_titles[4], fontsize=14, fontweight='bold', pad=20)
        
        # Add row labels - only axis title, rank title will be added separately
        axes[row_idx, 0].set_ylabel('Optimization Pack Power (kW)', 
                                   fontsize=12, labelpad=8)
    
    # Add separate row rank titles on the left side with better alignment
    row_positions = [0.77, 0.5, 0.23]  # Calculated positions to align with subplot centers
    for i, title in enumerate(rank_titles):
        # Add rank title to the left of the leftmost subplot
        fig.text(0.06, row_positions[i], title, fontsize=16, fontweight='bold', 
                rotation=90, va='center', ha='center')
    
    # Clean up axis tick labels - only show on outer edges
    for i in range(3):
        for j in range(5):
            ax = axes[i, j]
            
            # X-axis labels: only on bottom row AND top row
            if i == 2:  # Bottom row
                ax.set_xlabel('Optimization Pack Energy (kWh)')
            elif i == 0:  # Top row - add x-axis labels at top
                ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
                ax.xaxis.set_label_position('top')
                ax.set_xlabel('Optimization Pack Energy (kWh)')
            else:  # Middle row - no x labels
                ax.set_xticklabels([])
            
            # Y-axis labels: leftmost column AND rightmost column
            if j == 0:  # Leftmost column - keep default left labels
                pass  # Keep the existing ylabel from row labels
            elif j == 4:  # Rightmost column - add right labels
                ax.tick_params(right=True, labelright=True, left=True, labelleft=False)
                ax.yaxis.set_label_position('right')
                ax.set_ylabel('Optimization Pack Power (kW)', 
                             fontsize=12, labelpad=8)
            else:  # Middle columns - no y labels
                ax.set_yticklabels([])
    
    # Add legend for cell types - make it more visible
    handles = [plt.Line2D([0], [0], marker='s', color='w', label=cell,
                         markerfacecolor=cell_colors[cell], markersize=15, markeredgecolor='k', markeredgewidth=2)
               for cell in sorted(all_cells)]
    
    # Position legend on the right side with more space to grow
    legend = fig.legend(handles=handles, title="Cell Type", loc='center left', 
                       bbox_to_anchor=(0.98, 0.75), frameon=True, 
                       fancybox=True, shadow=True, fontsize=9, 
                       title_fontsize=10, edgecolor='black', facecolor='white')
    legend.get_title().set_fontweight('bold')
    
    # Add colorbar for weights - moved down to give legend more space
    sm_weight = plt.cm.ScalarMappable(cmap=weight_cmap, norm=global_weight_norm)
    sm_weight.set_array([])
    cbar_weight_ax = fig.add_axes([0.98, 0.40, 0.012, 0.20])
    cbar_weight = fig.colorbar(sm_weight, cax=cbar_weight_ax, label='Pack Weight (kg)')
    cbar_weight.ax.tick_params(labelsize=8)
    
    # Add colorbar for weight differences - moved down further
    sm_diff = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=global_diff_norm)
    sm_diff.set_array([])
    cbar_diff_ax = fig.add_axes([0.98, 0.15, 0.012, 0.20])
    cbar_diff = fig.colorbar(sm_diff, cax=cbar_diff_ax, label='|Weight Diff| (kg)')
    cbar_diff.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.94, bottom=0.12, top=0.88, left=0.11, hspace=0.4, wspace=0.2)
    
    # Save plot to specified folder or current directory
    if output_folder:
        filepath = os.path.join(output_folder, f"comprehensive_analysis{UDT}.png")
    else:
        filepath = f"comprehensive_analysis{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Comprehensive analysis saved as {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_executive_summary(resultsCylDict, resultsAllDict, show_plot=True, UDT="", output_folder=None):
    """
    Simple 2x2 dashboard for executives/newcomers showing key insights at a glance
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for analysis
    all_cells = {}
    cyl_cells = {}
    
    # Process all cells data
    for key, result_data in resultsAllDict.items():
        for config in result_data['results']:
            if config is not None:
                cell = normalize_cell_name(config.iloc[Locations["Cell"]])
                weight = config.iloc[Locations["Total Pack Weight (kg)"]]
                if cell not in all_cells:
                    all_cells[cell] = []
                all_cells[cell].append(weight)
    
    # Process cylindrical data
    for key, result_data in resultsCylDict.items():
        for config in result_data['results']:
            if config is not None:
                cell = normalize_cell_name(config.iloc[Locations["Cell"]])
                weight = config.iloc[Locations["Total Pack Weight (kg)"]]
                if cell not in cyl_cells:
                    cyl_cells[cell] = []
                cyl_cells[cell].append(weight)
    
    # Top-left: Cell selection frequency (pie chart)
    ax = axes[0, 0]
    cell_counts = {cell: len(weights) for cell, weights in all_cells.items()}
    colors = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#8B00FF', '#00FFFF', '#FF1493']
    wedges, texts, autotexts = ax.pie(cell_counts.values(), labels=cell_counts.keys(), 
                                     autopct='%1.1f%%', colors=colors[:len(cell_counts)])
    ax.set_title('Cell Type Selection Frequency\n(All Configurations)', fontsize=14, fontweight='bold')
    
    # Top-right: Average weight by cell type (bar chart)
    ax = axes[0, 1]
    cell_names = list(all_cells.keys())
    avg_weights = [np.mean(all_cells[cell]) for cell in cell_names]
    bars = ax.bar(cell_names, avg_weights, color=colors[:len(cell_names)])
    ax.set_title('Average Pack Weight by Cell Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Weight (kg)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, weight in zip(bars, avg_weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{weight:.1f}', ha='center', va='bottom')
    
    # Bottom-left: Pack Power vs Pack Energy colored by cell type (like comprehensive analysis top-right)
    ax = axes[1, 0]
    
    # Extract optimal solutions data for Power vs Energy visualization
    cell_colors_map = {}
    color_idx = 0
    
    # Plot each optimal configuration as a point colored by cell type
    for key, result_data in resultsAllDict.items():
        if result_data['results'] and result_data['results'][0] is not None:
            optimal_config = result_data['results'][0]
            cell = normalize_cell_name(optimal_config.iloc[Locations["Cell"]])
            pack_power = optimal_config.iloc[Locations["Pack Power (kW)"]]
            pack_energy = optimal_config.iloc[Locations["Pack Energy (kWh)"]]
            
            # Assign consistent color to each cell type
            if cell not in cell_colors_map:
                cell_colors_map[cell] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = cell_colors_map[cell]
            ax.scatter(pack_energy, pack_power, c=color, s=30, alpha=0.8, 
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Pack Energy (kWh)')
    ax.set_ylabel('Pack Power (kW)')
    ax.set_title('Optimal Configurations\n(Pack Power vs Energy by Cell Type)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend for cell types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=8, label=cell, markeredgecolor='black')
                      for cell, color in cell_colors_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # Bottom-right: Key metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate key metrics
    total_configs = sum(len(result_data['results']) for result_data in resultsAllDict.values())
    best_cell = max(cell_counts.items(), key=lambda x: x[1])[0]
    lightest_avg = min(avg_weights)
    heaviest_avg = max(avg_weights)
    
    # Calculate optimal solution statistics
    optimal_count = len([key for key, result_data in resultsAllDict.items() 
                        if result_data['results'] and result_data['results'][0] is not None])
    
    metrics_text = f"""
    KEY INSIGHTS:
    
    • Total Configurations: {total_configs:,}
    • Optimal Solutions Found: {optimal_count:,}
    • Most Selected Cell: {best_cell}
    • Lightest Average: {lightest_avg:.1f} kg
    • Heaviest Average: {heaviest_avg:.1f} kg
    • Cell Types Available: {len(cell_names)}
    
    RECOMMENDATION:
    Use {best_cell} for most applications
    """
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    
    # Save plot
    if output_folder:
        filepath = os.path.join(output_folder, f"executive_summary{UDT}.png")
    else:
        filepath = f"executive_summary{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Executive summary saved as {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_cell_ranking(resultsAllDict, show_plot=True, UDT="", output_folder=None):
    """
    Vertical bar chart showing cell selection frequency by ranking position (1st, 2nd, 3rd choice)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data and track ranking positions
    cell_rankings = {}
    total_configurations = 0
    
    for key, result_data in resultsAllDict.items():
        total_configurations += 1
        for rank, config in enumerate(result_data['results'][:3]):  # Top 3 choices
            if config is not None:
                cell = normalize_cell_name(config.iloc[Locations["Cell"]])
                if cell not in cell_rankings:
                    cell_rankings[cell] = {'first': 0, 'second': 0, 'third': 0}
                
                if rank == 0:
                    cell_rankings[cell]['first'] += 1
                elif rank == 1:
                    cell_rankings[cell]['second'] += 1
                elif rank == 2:
                    cell_rankings[cell]['third'] += 1
    
    # Calculate percentages and sort by total frequency
    cell_data = []
    for cell, rankings in cell_rankings.items():
        total_selections = rankings['first'] + rankings['second'] + rankings['third']
        first_pct = (rankings['first'] / total_configurations) * 100
        second_pct = (rankings['second'] / total_configurations) * 100
        third_pct = (rankings['third'] / total_configurations) * 100
        cell_data.append((cell, first_pct, second_pct, third_pct, total_selections))
    
    # Sort by total selections (descending)
    cell_data.sort(key=lambda x: x[4], reverse=True)
    
    # Unpack data
    cell_names = [item[0] for item in cell_data]
    first_pct = [item[1] for item in cell_data]
    second_pct = [item[2] for item in cell_data]
    third_pct = [item[3] for item in cell_data]
    
    # Create stacked bar chart
    x_pos = np.arange(len(cell_names))
    
    bars1 = ax.bar(x_pos, first_pct, label='1st Choice', color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x_pos, second_pct, bottom=first_pct, label='2nd Choice', color='#FFA500', alpha=0.8)
    bars3 = ax.bar(x_pos, third_pct, bottom=np.array(first_pct) + np.array(second_pct), 
                   label='3rd Choice', color='#DC143C', alpha=0.8)
    
    # Add percentage labels on bars
    for i, (cell, f_pct, s_pct, t_pct, total) in enumerate(cell_data):
        # Label for 1st choice (if significant)
        if f_pct > 2:  # Only show if > 2%
            ax.text(i, f_pct/2, f'{f_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
        
        # Label for 2nd choice (if significant)
        if s_pct > 2:
            ax.text(i, f_pct + s_pct/2, f'{s_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
        
        # Label for 3rd choice (if significant)
        if t_pct > 2:
            ax.text(i, f_pct + s_pct + t_pct/2, f'{t_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=9)
        
        # Total count above bar
        total_pct = f_pct + s_pct + t_pct
        if total_pct > 0:
            ax.text(i, total_pct + 0.5, f'{total}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Selection Frequency (%)', fontsize=12)
    ax.set_title('Cell Type Selection Frequency by Ranking Position\n(Percentage of total configurations)', 
                fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cell_names, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with summary
    summary_text = f"Total Configurations: {total_configurations:,}\nCells Analyzed: {len(cell_names)}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_folder:
        filepath = os.path.join(output_folder, f"cell_ranking{UDT}.png")
    else:
        filepath = f"cell_ranking{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Cell ranking saved as {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_tradeoffs(resultsAllDict, show_plot=True, UDT="", output_folder=None):
    """
    Scatter plot showing Energy Density vs Power Density trade-offs
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Extract data
    cell_data = {}
    for key, result_data in resultsAllDict.items():
        for config in result_data['results']:
            if config is not None:
                cell = normalize_cell_name(config.iloc[Locations["Cell"]])
                energy_density = config.iloc[Locations["Grav Energy Density (Wh/kg)"]]
                power_density = config.iloc[Locations["Grav Power Density (W/kg)"]]
                weight = config.iloc[Locations["Total Pack Weight (kg)"]]
                
                if cell not in cell_data:
                    cell_data[cell] = {'energy_density': [], 'power_density': [], 'weights': []}
                
                cell_data[cell]['energy_density'].append(energy_density)
                cell_data[cell]['power_density'].append(power_density)
                cell_data[cell]['weights'].append(weight)
    
    # Plot each cell type
    colors = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#8B00FF', '#00FFFF', '#FF1493']
    
    for i, (cell, data) in enumerate(cell_data.items()):
        # Calculate averages for positioning
        avg_energy_density = np.mean(data['energy_density'])
        avg_power_density = np.mean(data['power_density'])
        frequency = len(data['weights'])
        avg_weight = np.mean(data['weights'])
        
        # More exaggerated size based on selection frequency
        size = (frequency ** 1.5) * 10  # Exaggerated scaling with power function
        color = colors[i % len(colors)]
        
        scatter = ax.scatter(avg_energy_density, avg_power_density, s=size, c=color, 
                           alpha=0.7, edgecolors='black', linewidth=1, label=cell)
        
        # Add text labels
        ax.annotate(f'{cell}\n({avg_weight:.1f}kg)', 
                   (avg_energy_density, avg_power_density),
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Gravimetric Energy Density (Wh/kg)', fontsize=12)
    ax.set_ylabel('Gravimetric Power Density (W/kg)', fontsize=12)
    ax.set_title('Cell Performance Trade-offs\n(Bubble size = Selection frequency, Labels show average weight)', 
                fontsize=14, fontweight='bold')
    
    # Use tight axes instead of starting at 0
    ax.autoscale(tight=True)
    
    # Add grid for easier reading
    ax.grid(True, alpha=0.3)
    
    # Add legend with uniform bubble sizes
    legend_elements = []
    for i, cell in enumerate(cell_data.keys()):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=8, 
                                        label=cell, markeredgecolor='black', markeredgewidth=1))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    if output_folder:
        filepath = os.path.join(output_folder, f"tradeoffs{UDT}.png")
    else:
        filepath = f"tradeoffs{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Trade-offs analysis saved as {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_improvement_potential(resultsCylDict, resultsAllDict, show_plot=True, UDT="", output_folder=None):
    """
    Before/After comparison showing cylindrical vs optimized performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    cyl_weights = []
    all_weights = []
    power_energy_pairs = []
    
    for key in resultsCylDict.keys():
        if key in resultsAllDict:
            cyl_config = resultsCylDict[key]['results'][0]  # Best cylindrical
            all_config = resultsAllDict[key]['results'][0]  # Best all cells
            
            if cyl_config is not None and all_config is not None:
                cyl_weight = cyl_config.iloc[Locations["Total Pack Weight (kg)"]]
                all_weight = all_config.iloc[Locations["Total Pack Weight (kg)"]]
                
                cyl_weights.append(cyl_weight)
                all_weights.append(all_weight)
                power_energy_pairs.append(key)
    
    # Top-left: Weight comparison scatter
    ax = axes[0, 0]
    ax.scatter(cyl_weights, all_weights, alpha=0.6)
    
    # Add diagonal line for reference
    min_weight = min(min(cyl_weights), min(all_weights))
    max_weight = max(max(cyl_weights), max(all_weights))
    ax.plot([min_weight, max_weight], [min_weight, max_weight], 'r--', alpha=0.8, label='No improvement')
    
    ax.set_xlabel('Cylindrical Cell Weight (kg)')
    ax.set_ylabel('Optimized Cell Weight (kg)')
    ax.set_title('Weight Comparison: Cylindrical vs Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Weight savings histogram
    ax = axes[0, 1]
    weight_savings = [cyl - all for cyl, all in zip(cyl_weights, all_weights)]
    ax.hist(weight_savings, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(weight_savings), color='red', linestyle='--', linewidth=2, 
               label=f'Mean savings: {np.mean(weight_savings):.1f}kg')
    ax.set_xlabel('Weight Savings (kg)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Weight Savings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Percentage improvement
    ax = axes[1, 0]
    percent_improvements = [(cyl - all)/cyl * 100 for cyl, all in zip(cyl_weights, all_weights)]
    ax.hist(percent_improvements, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(percent_improvements), color='red', linestyle='--', linewidth=2, 
               label=f'Mean improvement: {np.mean(percent_improvements):.1f}%')
    ax.set_xlabel('Weight Reduction (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Percentage Weight Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    IMPROVEMENT SUMMARY:
    
    • Configurations analyzed: {len(weight_savings)}
    • Average weight savings: {np.mean(weight_savings):.1f} kg
    • Maximum weight savings: {max(weight_savings):.1f} kg
    • Average improvement: {np.mean(percent_improvements):.1f}%
    • Maximum improvement: {max(percent_improvements):.1f}%
    • Configurations improved: {sum(1 for x in weight_savings if x > 0)}
    • Improvement rate: {sum(1 for x in weight_savings if x > 0)/len(weight_savings)*100:.1f}%
    
    RECOMMENDATION:
    Switching from cylindrical-only to 
    optimized cell selection can save 
    {np.mean(weight_savings):.1f}kg on average
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    
    plt.tight_layout()
    
    # Save plot
    if output_folder:
        filepath = os.path.join(output_folder, f"improvement_potential{UDT}.png")
    else:
        filepath = f"improvement_potential{UDT}.png"
    
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    print(f"Improvement potential analysis saved as {filepath}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # Load results from pickle file
    Density = 5  # Default density, can be changed
    
    # Check for existing results files
    possible_files = [f"results{Density}.pkl", "results100.pkl", "results500.pkl"]
    results_file = None
    
    for file in possible_files:
        if os.path.exists(file):
            results_file = file
            break
    
    if results_file is None:
        print("No results pickle file found. Looking for any results*.pkl files...")
        for file in os.listdir("."):
            if file.startswith("results") and file.endswith(".pkl"):
                results_file = file
                print(f"Found: {file}")
                break
    
    if results_file is None:
        print("Error: No results pickle file found. Please run the analysis first.")
        exit(1)
    
    # Extract density from filename
    try:
        density_str = results_file.replace("results", "").replace(".pkl", "")
        if density_str.isdigit():
            Density = int(density_str)
    except:
        pass
    
    print(f"Loading results from {results_file}...")
    try:
        resultsDict = pickle.load(open(results_file, "rb"))
        resultsCylDict, resultsAllDict = resultsDict[0], resultsDict[1]
        print(f"Loaded {len(resultsCylDict)} cylindrical results and {len(resultsAllDict)} all-cell results")
        
        # Generate timestamp and create folder
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f"plots_{timestamp}_density{Density}"
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        
        # Generate plots
        print("Generating cylindrical optimization plot...")
        plot_optimization_scatter_coords(resultsCylDict, show_plot=False, 
                                        UDT=f"_Cylindrical_D{Density}", 
                                        output_folder=folder_name)
        
        print("Generating all-cells optimization plot...")
        plot_optimization_scatter_coords(resultsAllDict, show_plot=False, 
                                        UDT=f"_All_D{Density}", 
                                        output_folder=folder_name)
        
        print("Generating comprehensive analysis plot...")
        plot_comprehensive_analysis(resultsCylDict, resultsAllDict, show_plot=False, 
                                   UDT=f"_D{Density}", 
                                   output_folder=folder_name)
        
        # Generate the new simplified plots for executives/newcomers
        print("Generating executive summary...")
        plot_executive_summary(resultsCylDict, resultsAllDict, show_plot=False, 
                              UDT=f"_summary_D{Density}", 
                              output_folder=folder_name)
        
        print("Generating cell ranking analysis...")
        plot_cell_ranking(resultsAllDict, show_plot=False, 
                         UDT=f"_ranking_D{Density}", 
                         output_folder=folder_name)
        
        print("Generating trade-offs analysis...")
        plot_tradeoffs(resultsAllDict, show_plot=False, 
                      UDT=f"_tradeoffs_D{Density}", 
                      output_folder=folder_name)
        
        print("Generating improvement potential analysis...")
        plot_improvement_potential(resultsCylDict, resultsAllDict, show_plot=False, 
                                  UDT=f"_improvements_D{Density}", 
                                  output_folder=folder_name)
        
        print(f"All plots generated successfully in folder: {folder_name}")
        
    except Exception as e:
        print(f"Error loading or processing results: {e}")
        exit(1)
