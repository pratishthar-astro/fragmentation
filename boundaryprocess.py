import numpy as np
import pandas as pd
import glob
import argparse

# Constants
low_density_fixed = 1e-15  # g/cm³
sigma_SB = 5.670374419e-5  # Stefan-Boltzmann constant in erg cm⁻² s⁻¹ K⁻⁴
L_sun = 3.828e33  # Solar luminosity in erg/s

# Simulation setup: location of files, name, particle mass, opacity scaling as in Meru (2015)
SIMULATIONS = {
    1: {"dir": "Scripts/New_rdump/", "file_pattern": "ptcl1_n*", "particle_mass": 4.549e-6, "opacity_scale": 0.9, "luminosity": 4.3 * L_sun},
    2: {"dir": "Scripts/New_rdump/", "file_pattern": "ptcl2_n*", "particle_mass": 3.695e-6, "opacity_scale": 0.5, "luminosity": 2.4 * L_sun}
}

def compute_T(R, luminosity):
    """Compute temperature using T ∝ R^(-1/2) and Stefan-Boltzmann law."""
    T = (luminosity / (4 * np.pi * sigma_SB * R**2))**(1/4)
    return T

def compute_opacity(temperature, density, scale_factor):
    """Compute opacity based on temperature and density using different opacity regimes."""
    opacity_regimes = [
        {"k0": 2e-4, "a": 0, "b": 2, "T_min": 0, "T_max": 1.6681e2},
        {"k0": 2e16, "a": 0, "b": -7, "T_min": 1.6681e2, "T_max": 2.02677e2},
        {"k0": 1e-1, "a": 0, "b": 1/2, "T_min": 2.02677e2, "T_max": 2.28677e3 * density**(2/49)},
        {"k0": 2e81, "a": 1, "b": -24, "T_min": 2.28677e3 * density**(2/49), "T_max": 2.02976e3 * density**(1/81)},
        {"k0": 1e-8, "a": 2/3, "b": 3, "T_min": 2.02976e3 * density**(1/81), "T_max": 1e4 * density**(1/21)},
        {"k0": 1e-36, "a": 1/3, "b": 10, "T_min": 1e4 * density**(1/21), "T_max": 3.11952e4 * density**(4/75)},
        {"k0": 1.5e20, "a": 1, "b": -5/2, "T_min": 3.11952e4 * density**(4/75), "T_max": 1.79393e8 * density**(2/5)},
        {"k0": 3.48e-1, "a": 0, "b": 0, "T_min": 1.79393e8 * density**(2/5), "T_max": np.inf}
    ]

    for regime in opacity_regimes:
        if regime["T_min"] <= temperature < regime["T_max"]:
            return regime["k0"] * (density ** regime["a"]) * (temperature ** regime["b"]) * scale_factor
    return np.nan  

def compute_boundary_height(df_bin, annulus_area, mean_opacity, particle_mass):
    """Compute boundary heights using the N_b-th particle method or the 10% boundary particle method."""
    if np.isnan(mean_opacity) or mean_opacity <= 0:
        return np.nan, np.nan  

    Nb = int(annulus_area / (mean_opacity * particle_mass))

    df_sorted_pos = df_bin[df_bin['z'] >= 0].sort_values(by='z', ascending=False)
    df_sorted_neg = df_bin[df_bin['z'] < 0].sort_values(by='z', ascending=True)

    # 1. Find the boundary height based on the N-th particle.
    z_boundary_pos_Nb = df_sorted_pos.iloc[Nb]['z'] if Nb < len(df_sorted_pos) else df_sorted_pos['z'].max()
    z_boundary_neg_Nb = df_sorted_neg.iloc[Nb]['z'] if Nb < len(df_sorted_neg) else df_sorted_neg['z'].min()

    # Compute the uppermost 10% particles and take the minimum z from that group.
    n_total = len(df_bin)
    n_boundary = int(0.1 * n_total)

    # For the positive z particles (uppermost 10%)
    df_sorted_pos_10 = df_sorted_pos.head(n_boundary)
    z_boundary_pos_10 = df_sorted_pos_10['z'].min() if not df_sorted_pos_10.empty else df_sorted_pos['z'].max()

    # For the negative z particles (uppermost 10%)
    df_sorted_neg_10 = df_sorted_neg.head(n_boundary)
    z_boundary_neg_10 = df_sorted_neg_10['z'].min() if not df_sorted_neg_10.empty else df_sorted_neg['z'].min()

    # 3. Compute the maximum of the two heights: N-th particle or 10% boundary
    z_boundary_pos = max(z_boundary_pos_Nb, z_boundary_pos_10)
    z_boundary_neg = max(z_boundary_neg_Nb, z_boundary_neg_10)

    return z_boundary_pos, z_boundary_neg

def process_simulation(sim_num, bin_width):
    """Reads simulation data, computes opacity & boundary height, and saves results."""
    sim = SIMULATIONS[sim_num]
    file_list = sorted(glob.glob(sim["dir"] + sim["file_pattern"]))
    
    if not file_list:
        print(f"No files found for Simulation {sim_num}.")
        return

    all_data = []

    for file in file_list:
        with open(file, 'r') as f:
            header = f.readline().strip()
            gt_value = float(header.split('=')[1])

        df = pd.read_csv(file, sep='\s+', skiprows=2, header=None,
                         names=['i', 'iunique', 'iphase', 'x', 'y', 'z', 'R', 'u', 'Cv', 'n', 'H'])

        df['temperature'] = compute_T(df['R'], sim["luminosity"])

        # Handle NaN temperatures by using the average temperature for the bin
        df['temperature'] = df.apply(lambda row: np.nan if row['Cv'] == 0 else row['temperature'], axis=1)
        df['realtime'] = gt_value

        # Replace NaN temperatures with the average temperature for the bin
        all_data.append(df)

    # Concatenate all data into a single DataFrame
    all_data = pd.concat(all_data, ignore_index=True)

    # Create radial_bin column after concatenating all data
    bin_edges = np.arange(all_data['R'].min(), all_data['R'].max() + bin_width, bin_width)
    bin_labels = (bin_edges[:-1] + bin_edges[1:]) / 2
    all_data['radial_bin'] = pd.cut(all_data['R'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Replace NaN temperatures with the average temperature of the bin
    for bin_center in all_data['radial_bin'].unique():
        bin_data = all_data[all_data['radial_bin'] == bin_center]
        avg_temp = bin_data['temperature'].mean()

        # Replace NaN values with the average temperature of the bin
        all_data.loc[(all_data['radial_bin'] == bin_center) & (all_data['temperature'].isna()), 'temperature'] = avg_temp

    # Compute opacity for each particle
    all_data['opacity'] = all_data.apply(lambda row: compute_opacity(row['temperature'], low_density_fixed, sim["opacity_scale"]), axis=1)

    # Bin data and compute boundary heights
    boundary_heights = {}
    for bin_center in bin_labels:
        df_bin = all_data[all_data['radial_bin'] == bin_center]
        if df_bin.empty:
            continue

        Rin, Rout = bin_center - bin_width / 2, bin_center + bin_width / 2
        annulus_area = np.pi * (Rout**2 - Rin**2)

        mean_opacity = df_bin['opacity'].mean()
        z_pos, z_neg = compute_boundary_height(df_bin, annulus_area, mean_opacity, sim["particle_mass"])
        boundary_heights[bin_center] = (z_pos, z_neg)

        print(f"Bin Center: {bin_center}, Boundary Heights: {z_pos}, {z_neg}")

    # Assign boundary height & particle flag
    all_data['boundary_height_pos'] = all_data['radial_bin'].map(lambda x: boundary_heights.get(x, (np.nan, np.nan))[0])
    all_data['boundary_height_neg'] = all_data['radial_bin'].map(lambda x: boundary_heights.get(x, (np.nan, np.nan))[1])
    all_data['boundary_flag'] = ((all_data['z'] >= all_data['boundary_height_pos']) | 
                                 (all_data['z'] <= all_data['boundary_height_neg'])).astype(int)

    all_data.to_csv(f"processed_data_sim{sim_num}.csv", index=False)
    print(f"Processed data saved as 'processed_data_sim{sim_num}.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess simulation data.")
    parser.add_argument("--sim", type=int, required=True, help="Simulation number (1 or 2).")
    parser.add_argument("--bin_width", type=float, default=0.1, help="Radial bin width (default: 0.1 AU).")
    args = parser.parse_args()

    process_simulation(args.sim, args.bin_width)

