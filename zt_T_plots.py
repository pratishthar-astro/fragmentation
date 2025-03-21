import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import warnings
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import os

warnings.filterwarnings("ignore")

output_dir = 'zT_plots'
os.makedirs(output_dir, exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sim', type=int, choices=[1, 2], required=True, help="Simulation number (1 or 2)")
args = parser.parse_args()

# Set filenames based on simulation
if args.sim == 1:
    df_files = ['Scripts/Adisc01_ascii', 'Scripts/Adisc02_ascii', 'Scripts/Adisc03_ascii']
    processed_data_file = 'processed_data_sim1.csv'
    firstfrag_id, secondfrag_id = 119369, 35936
    secondfrag_split_time = 9032
    output_filename = 'z_t_T_sim1_variable.png'
    selected_iuniques = [319, 74255]
else:
    df_files = ['Scripts/Adisc01_asciisim2', 'Scripts/Adisc02_asciisim2']
    processed_data_file = 'processed_data_sim2.csv'
    firstfrag_id, secondfrag_id = 63589, 188311
    secondfrag_split_time = 13832.8
    output_filename = 'z_t_T_sim2.png'
    selected_iuniques = [74995, 237001]

# Read the data files
df_list = [pd.read_csv(f, delimiter='\s+', header=None) for f in df_files]
df_concat = pd.concat(df_list, ignore_index=True)
df_concat.columns = ['iunique', 'realtime', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'iuniquelocal']
df_concat['realtime_years'] = df_concat['realtime'] * 0.159
df_concat['Radius'] = np.sqrt(df_concat['x']**2 + df_concat['y']**2)

# Separate relevant fragments
df_firstfrag = df_concat[df_concat['iuniquelocal'] == firstfrag_id]
df_secondfrag = df_concat[df_concat['iuniquelocal'] == secondfrag_id]

df_secondfrag1 = df_secondfrag[df_secondfrag['realtime'] <= secondfrag_split_time]
df_secondfrag2 = df_secondfrag[df_secondfrag['realtime'] > secondfrag_split_time]

# Read processed data
all_data = pd.read_csv(processed_data_file)

df_1030_b = all_data[all_data['iunique'].isin(df_secondfrag1['iunique'])]
df_1030_a = all_data[all_data['iunique'].isin(df_secondfrag2['iunique'])]

print(df_1030_b.columns)
print(df_1030_b.shape)  # Should return (rows, columns)
print(df_1030_b.head())  # See if it contains data


df_1030_b['realtime_years'] = df_1030_b['realtime'] * 0.159
df_1030_b['T'] = df_1030_b['u'] / df_1030_b['Cv']

# Compute the normalized height
min_boundary_height = df_1030_b[['boundary_height_pos', 'boundary_height_neg']].abs().min(axis=1)
df_1030_b['normalized_z'] = df_1030_b['z'] / min_boundary_height

# Identify boundary-exceeding particles
df_secondfrag_bpex = df_1030_b[df_1030_b['n'] == 0]
print(f"iuniques: {len(df_secondfrag_bpex['iunique'].unique())}")
num_particles_above_boundary = ((df_secondfrag_bpex['normalized_z'] >= 1) | (df_secondfrag_bpex['normalized_z'] <= -1)).sum()
print(f"Total particles exceeding boundary: {num_particles_above_boundary}")
print(f"Total particles: {len(df_secondfrag_bpex['iunique'].unique())}")

# Plot setup
num_plots = len(df_secondfrag_bpex['iunique'].dropna().unique())
plots_per_row = 4
num_rows = num_plots // (2 * plots_per_row) + (1 if num_plots % (2 * plots_per_row) else 0)

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(18, 6 * num_rows))
axes = axes.flatten()

iunique_values = df_secondfrag_bpex['iunique'].dropna().unique()

for idx in range(0, num_plots, 2):
    ax = axes[idx // 2]
    iunique_val1 = iunique_values[idx]
    iunique_val2 = iunique_values[idx + 1] if idx + 1 < num_plots else None
    
    subset1 = df_secondfrag_bpex[df_secondfrag_bpex['iunique'] == iunique_val1]
    ax.scatter(subset1['realtime'], subset1['normalized_z'], c=subset1['T'], cmap='viridis', alpha=0.7)
    ax.plot(subset1['realtime'], subset1['normalized_z'], c='black', alpha=0.5)
    
    if iunique_val2:
        subset2 = df_secondfrag_bpex[df_secondfrag_bpex['iunique'] == iunique_val2]
        ax.scatter(subset2['realtime'], subset2['normalized_z'], c=subset2['T'], cmap='viridis', alpha=0.7)
        ax.plot(subset2['realtime'], subset2['normalized_z'], c='black', alpha=0.5)
    
    ax.axhline(y=1, color='r', linestyle='--', label='Boundary Height')
    ax.axhline(y=-1, color='r', linestyle='--', label='Boundary Height')
    ax.set_xlabel('Realtime')
    ax.set_ylabel('Z / Min Boundary Height')
    ax.set_title(f'Z/Boundary vs Realtime for iunique={iunique_val1}' + (f' and {iunique_val2}' if iunique_val2 else ''))
    ax.legend()
    
    if (idx // 2 + 1) % plots_per_row == 0:
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('T')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
plt.show()

# For 3D plotting
num_plots = len(df_secondfrag_bpex['iunique'].dropna().unique())
plots_per_row = 4
num_rows = num_plots // (2 * plots_per_row) + (1 if num_plots % (2 * plots_per_row) else 0)

fig = plt.figure(figsize=(18, 6 * num_rows))

iunique_values = df_secondfrag_bpex['iunique'].dropna().unique()

for idx in range(0, num_plots, 2):
    row_idx = idx // 2
    ax = fig.add_subplot(num_rows, plots_per_row, row_idx + 1, projection='3d')

    iunique_val1 = iunique_values[idx]
    iunique_val2 = iunique_values[idx + 1] if idx + 1 < num_plots else None
    
    # --- First iunique ---
    subset1 = df_secondfrag_bpex[df_secondfrag_bpex['iunique'] == iunique_val1].sort_values('realtime')
    sc1 = ax.scatter(subset1['normalized_z'], subset1['realtime'], subset1['R'], 
                     c=subset1['T'], cmap='viridis', alpha=0.7, label=f'iunique={iunique_val1}')
    
    # Line segments for iunique1
    points1 = subset1[['normalized_z', 'realtime', 'R']].values.reshape(-1, 1, 3)
    segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
    norm = mcolors.Normalize(vmin=subset1['T'].min(), vmax=subset1['T'].max())
    lc1 = Line3DCollection(segments1, cmap='viridis', norm=norm, linewidth=1.5, alpha=0.7)
    lc1.set_array(subset1['T'].values[:-1])  # Set color values
    ax.add_collection(lc1)
    
    # --- Second iunique (if available) ---
    if iunique_val2:
        subset2 = df_secondfrag_bpex[df_secondfrag_bpex['iunique'] == iunique_val2].sort_values('realtime')
        sc2 = ax.scatter(subset2['normalized_z'], subset2['realtime'], subset2['R'], 
                         c=subset2['T'], cmap='viridis', alpha=0.7, label=f'iunique={iunique_val2}')
        
        points2 = subset2[['normalized_z', 'realtime', 'R']].values.reshape(-1, 1, 3)
        segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
        lc2 = Line3DCollection(segments2, cmap='viridis', norm=norm, linewidth=1.5, alpha=0.7)
        lc2.set_array(subset2['T'].values[:-1])
        ax.add_collection(lc2)

    ax.set_xlabel('z / Boundary Height')
    ax.set_ylabel('Realtime')
    ax.set_zlabel('R [AU]')
    ax.set_title(f'3D view: iunique={iunique_val1}' + (f' & {iunique_val2}' if iunique_val2 else ''))
    ax.legend()
    
    if (row_idx + 1) % plots_per_row == 0 or idx + 2 >= num_plots:
        cbar = plt.colorbar(sc1, ax=ax, shrink=0.6)
        cbar.set_label('T')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"3D_particle_multi_panel_sim{args.sim}.png"), dpi=300, bbox_inches='tight')
plt.show()

# Now check temperature changes
# Identify particles that have ever exceeded the boundary
particles_that_exceeded = df_1030_b[(df_1030_b['normalized_z'] >= 1) | (df_1030_b['normalized_z'] <= -1)]['iunique'].unique()

# Store temperature differences per particle
temp_diffs = []
for particle in particles_that_exceeded:
    particle_data = df_1030_b[df_1030_b['iunique'] == particle].sort_values('realtime')
    
    before = particle_data[particle_data['normalized_z'] < 1]
    after = particle_data[(particle_data['normalized_z'] >= 1) | (particle_data['normalized_z'] <= -1)]
    
    if not before.empty and not after.empty:
        temp_diffs.append({'iunique': particle, 'T_before': before.iloc[-1]['T'], 'T_after': after.iloc[0]['T']})

df_temp_diffs = pd.DataFrame(temp_diffs)
if df_temp_diffs.empty:
    print("No valid temperature differences found.")
    exit()

df_temp_diffs = df_temp_diffs.replace([np.inf, -np.inf], np.nan).dropna()

# Define filenames
scatter_filename = f"T_before_vs_after_sim{args.sim}.png"
hist_filename = f"T_change_hist_sim{args.sim}.png"

# Scatter plot: Temperature before vs after crossing
plt.figure(figsize=(8,6))
plt.scatter(df_temp_diffs['T_before'], df_temp_diffs['T_after'], alpha=0.6, color='blue')
plt.plot([0,1300], [0, 1300], '--', color='red')
plt.xlabel("T Before Crossing Boundary (K)")
plt.ylabel("T After Crossing Boundary (K)")
plt.xlim(0,1300)
plt.ylim(0,1300)
plt.title(f"Temperature Before vs After Boundary Interaction (Sim {args.sim})")
plt.savefig(os.path.join(output_dir, scatter_filename), dpi=300, bbox_inches='tight')
plt.show()

# Calculate percentage of points below y = x
below_line = (df_temp_diffs['T_after'] < df_temp_diffs['T_before']).sum()
total_particles = len(df_temp_diffs)
percentage_below = (below_line / total_particles) * 100
print(f"Cooling percent: {percentage_below}")

# Histogram of temperature changes
plt.figure(figsize=(8,6))
plt.hist(df_temp_diffs['T_after'] - df_temp_diffs['T_before'], bins=30, alpha=0.7, color='purple')
plt.axvline(x=0, linestyle="--", color="red")
plt.xlabel("ΔT")
plt.ylabel("Number of Particles")
plt.title(f"Temperature Change Before vs After Boundary Interaction (Sim {args.sim})")
plt.savefig(os.path.join(output_dir, hist_filename), dpi=300, bbox_inches='tight')
plt.show()

# checking and comparing iuniques (the ones that cool several times after interaction and ones that don't)
# visually inspected
df_target_particles = df_secondfrag_bpex[df_secondfrag_bpex['iunique'].isin(selected_iuniques)]
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['multiple interactions', 'one interaction']

for idx, iunique_val in enumerate(selected_iuniques):
    subset = df_secondfrag_bpex[df_secondfrag_bpex['iunique'] == iunique_val]
    ax.plot(subset['realtime'], subset['R'], linestyle='-', color=f'C{idx}', alpha=0.6, label=f'{labels[idx]} (line)')
    sc = ax.scatter(subset['realtime'], subset['R'] , c=subset['T'], cmap='viridis', alpha=0.7)

ax.set_ylabel('Radius [AU]')
ax.set_xlabel('Realtime')
ax.set_title(f'Radius vs Realtime colored by T for simulation {args.sim}')
ax.legend()

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('T (K)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"R_t_T_sim{args.sim}.png"), dpi=300, bbox_inches='tight')
plt.show()
