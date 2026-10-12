# Load and pool all barrage vs CA1/CA2/CA3 firing rate data
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Define directory with saved pkl files
save_dir = r"D:\Documents\GitHub\Sleep_criticaltiy\data\rats\barr_CA2_CA3"

# Find all pkl files
pkl_files = glob(os.path.join(save_dir, "*.pkl"))
print(f"Found {len(pkl_files)} pkl files")
print("="*60)

# Load all dataframes
all_dfs = []
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    filename = os.path.basename(pkl_file)
    print(f"Loaded: {filename} - Shape: {df.shape}")
    all_dfs.append(df)

print("\n" + "="*60)
print(f"Total files loaded: {len(all_dfs)}")
print("="*60)

# Define columns to analyze
fr_columns = [
    'CA1_E_FR_per_unit', 'CA1_I_FR_per_unit', 'CA1_All_FR_per_unit',
    'CA2_E_FR_per_unit', 'CA2_I_FR_per_unit', 'CA2_All_FR_per_unit',
    'CA3_E_FR_per_unit', 'CA3_I_FR_per_unit', 'CA3_All_FR_per_unit'
]

# Number of tiles
n_tiles = 6

# Function to compute FORCED sextile statistics for one file
def compute_sextile_stats(df, fr_column, n_tiles=6):
    """
    Compute FORCED sextile statistics for one file using percentile-based binning
    This ensures equal number of data points in each tile
    Returns tile means (x and y) and tile counts
    """
    # Filter non-zero barrage rate
    df_filtered = df[df['barrage_rate'] > 0].copy()

    if len(df_filtered) == 0:
        return None

    # Get data
    x = df_filtered['barrage_rate'].values
    y = df_filtered[fr_column].values

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0:
        return None

    # FORCED SEXTILE: Create tiles based on barrage rate percentiles
    # This ensures approximately equal sample sizes in each tile
    percentiles = np.linspace(0, 100, n_tiles + 1)
    tile_edges = np.percentile(x_clean, percentiles)

    # Compute mean for each tile
    tile_means_x = []
    tile_means_y = []
    tile_counts = []

    for i in range(n_tiles):
        mask_tile = (x_clean >= tile_edges[i]) & (x_clean < tile_edges[i+1])
        if i == n_tiles - 1:  # Include right edge for last tile
            mask_tile = (x_clean >= tile_edges[i]) & (x_clean <= tile_edges[i+1])

        if np.sum(mask_tile) > 0:
            tile_means_x.append(np.mean(x_clean[mask_tile]))
            tile_means_y.append(np.mean(y_clean[mask_tile]))
            tile_counts.append(np.sum(mask_tile))
        else:
            # This should rarely happen with percentile-based binning
            tile_means_x.append(np.nan)
            tile_means_y.append(np.nan)
            tile_counts.append(0)

    return {
        'tile_means_x': np.array(tile_means_x),
        'tile_means_y': np.array(tile_means_y),
        'tile_counts': np.array(tile_counts),
        'n_points': len(x_clean)
    }

# Compute sextile statistics for each file and each FR column
print("\n" + "="*60)
print("Computing sextile statistics for each file...")
print("="*60)

pooled_data = {col: [] for col in fr_columns}

for idx, df in enumerate(all_dfs):
    filename = os.path.basename(pkl_files[idx])
    print(f"\nProcessing: {filename}")

    for col in fr_columns:
        if col in df.columns:
            stats = compute_sextile_stats(df, col, n_tiles=n_tiles)
            if stats is not None:
                pooled_data[col].append(stats)
                print(f"  {col}: {stats['n_points']} points, tile counts: {stats['tile_counts']}")
        else:
            print(f"  {col}: Column not found")

# Compute pooled statistics (mean and SEM across files for each tile)
print("\n" + "="*60)
print("Computing pooled statistics...")
print("="*60)

pooled_stats = {}

for col in fr_columns:
    if len(pooled_data[col]) == 0:
        print(f"{col}: No data")
        continue

    # Collect all tile means across files
    all_tile_means_x = []
    all_tile_means_y = []

    for stats in pooled_data[col]:
        all_tile_means_x.append(stats['tile_means_x'])
        all_tile_means_y.append(stats['tile_means_y'])

    # Convert to arrays (files x tiles)
    all_tile_means_x = np.array(all_tile_means_x)
    all_tile_means_y = np.array(all_tile_means_y)

    # Compute mean and SEM across files for each tile
    pooled_mean_x = np.nanmean(all_tile_means_x, axis=0)
    pooled_mean_y = np.nanmean(all_tile_means_y, axis=0)
    pooled_sem_y = np.nanstd(all_tile_means_y, axis=0) / np.sqrt(np.sum(~np.isnan(all_tile_means_y), axis=0))

    pooled_stats[col] = {
        'mean_x': pooled_mean_x,
        'mean_y': pooled_mean_y,
        'sem_y': pooled_sem_y,
        'n_files': len(pooled_data[col])
    }

    print(f"{col}: Pooled from {len(pooled_data[col])} files")

# Create pooled plots
print("\n" + "="*60)
print("Creating pooled plots...")
print("="*60)

# Create 3x3 plot
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Pooled Barrage Rate vs CA1/CA2/CA3 Firing Rates (FORCED Sextile - Cross-file Mean ± SEM)',
             fontsize=16, fontweight='bold')

plot_configs = [
    ('CA1_E_FR_per_unit', 'CA1 Excitatory', 0, 0),
    ('CA1_I_FR_per_unit', 'CA1 Interneurons', 0, 1),
    ('CA1_All_FR_per_unit', 'CA1 All Cells', 0, 2),
    ('CA2_E_FR_per_unit', 'CA2 Excitatory', 1, 0),
    ('CA2_I_FR_per_unit', 'CA2 Interneurons', 1, 1),
    ('CA2_All_FR_per_unit', 'CA2 All Cells', 1, 2),
    ('CA3_E_FR_per_unit', 'CA3 Excitatory', 2, 0),
    ('CA3_I_FR_per_unit', 'CA3 Interneurons', 2, 1),
    ('CA3_All_FR_per_unit', 'CA3 All Cells', 2, 2),
]

for col_name, title, row, col in plot_configs:
    ax = axes[row, col]

    if col_name not in pooled_stats:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue

    stats = pooled_stats[col_name]

    # Remove NaN values
    valid_mask = ~(np.isnan(stats['mean_x']) | np.isnan(stats['mean_y']))
    x_plot = stats['mean_x'][valid_mask]
    y_plot = stats['mean_y'][valid_mask]
    sem_plot = stats['sem_y'][valid_mask]

    if len(x_plot) == 0:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        continue

    # Plot pooled means with error bars
    ax.errorbar(x_plot, y_plot, yerr=sem_plot,
                fmt='o-', color='darkblue', linewidth=2.5, markersize=10,
                capsize=6, capthick=2.5, label=f'Pooled (n={stats["n_files"]} files)')

    # Labels and title
    ax.set_xlabel('Barrage Rate (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{title} (Hz/unit)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure
output_dir = r"D:\Documents\GitHub\Sleep_criticaltiy\reports\figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "pooled_barrage_vs_CA123_firing_rates_forced_sextile.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved figure to: {output_path}")

output_path_svg = os.path.join(output_dir, "pooled_barrage_vs_CA123_firing_rates_forced_sextile.svg")
plt.savefig(output_path_svg, bbox_inches='tight')
print(f"✅ Saved figure to: {output_path_svg}")

plt.show()

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
