import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import imageio.v2 as imageio

def create_animation(df, output_folder, sim_num):
    """Create an animation of z vs R over time and save it as a GIF."""
    time_steps = sorted(df["realtime"].unique())

    images = []
    for t in time_steps:
        df_t = df[df["realtime"] == t]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(df_t["R"], df_t["z"], s=2, color="black", label="Normal Particles")
        plt.scatter(df_t[df_t["boundary_flag"] == 1]["R"], df_t[df_t["boundary_flag"] == 1]["z"], 
                    s=2, color="red", label="Boundary Particles")
        
        plt.xlim(0,80)
        plt.ylim(-40,40)
        plt.xlabel("Radial Distance R (AU)")
        plt.ylabel("Height z (AU)")
        plt.title(f"z vs R (Simulation {sim_num})\nTime: {t:.2f}")
        plt.legend()

        
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save frame
        frame_path = f"{output_folder}/frame_{t:.2f}.png"
        plt.savefig(frame_path, dpi=300)
        images.append(imageio.imread(frame_path))
        plt.close()

    # Create GIF
    gif_path = f"{output_folder}/animation_sim{sim_num}.gif"
    imageio.mimsave(gif_path, images, fps=5)
    
    print(f"Animation saved as {gif_path}.")

    # Cleanup: remove frame images after use
    for t in time_steps:
        frame_path = f"{output_folder}/frame_{t:.2f}.png"  # Ensure the correct path is used
        if os.path.exists(frame_path):
            os.remove(frame_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animation for simulation data.")
    parser.add_argument("--sim", type=int, required=True, help="Simulation number (1 or 2).")
    args = parser.parse_args()

    file_path = f"processed_data_sim{args.sim}.csv"
    output_folder = "plots"
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(file_path):
        print(f"Error: Processed data file {file_path} not found. Run `process_simulation.py` first.")
    else:
        df = pd.read_csv(file_path)
        create_animation(df, output_folder, args.sim)