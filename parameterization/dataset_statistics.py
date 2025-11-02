from MeshVolume import *
from pathlib import Path
import numpy as np
import os
import argparse
import csv # Import the csv module

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Statistics and Volume Estimation")
    parser.add_argument("--visualize", action="store_true", help="Visualize the reconstruction results")
    parser.add_argument("--store_results", action="store_true", help="Store the reconstruction results to files")
    parser.add_argument("--results_folder", type=str, default="./results", help="Folder to store the results")
    parser.add_argument("--start_index", type=int, default=2, help="Start index of the dataset to process")
    parser.add_argument("--end_index", type=int, default=10, help="End index of the dataset to process")   
    return parser.parse_args()

def calculate_error_pct(estimated, ground_truth):
    """Safely calculates the absolute percentage error."""
    if estimated is None or ground_truth is None or ground_truth == 0:
        return None
    return abs(estimated - ground_truth) / ground_truth * 100

def main():
    """
    Iterate through the dataset. For each object, run the volume estimation. 
    """

    # Read the txt file where all watertight mesh is stored
    dataset_txt_file = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/watertight_files.txt"
    with open(dataset_txt_file, "r") as f:
        mesh_files = f.readlines()[args.start_index:args.end_index] # Skip the first two lines (comments)

    # --- CSV Setup ---
    csv_writer = None
    csv_file = None
    if args.store_results:
        # Create the results directory if it doesn't exist
        os.makedirs(args.results_folder, exist_ok=True)
        csv_path = os.path.join(args.results_folder, "volume_estimation_results.csv")
        print(f"Storing results in {csv_path}")
        
        # Open the file and create a writer
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Define and write the header
        header = [
            'object_folder', 'gt_volume', 
            'psr_depth_3', 'psr_depth_3_error_pct',
            'psr_depth_4', 'psr_depth_4_error_pct',
            'psr_depth_5', 'psr_depth_5_error_pct',
            'psr_depth_6', 'psr_depth_6_error_pct',
            'psr_depth_7', 'psr_depth_7_error_pct',
            'sphere_ransac_mean', 'sphere_ransac_mean_error_pct', 'sphere_ransac_var',
            'cuboid_ransac_mean', 'cuboid_ransac_mean_error_pct', 'cuboid_ransac_var'
        ]
        csv_writer.writerow(header)
    # --- End CSV Setup ---

    for i, mesh in enumerate(mesh_files):
        mesh_path = mesh.strip()
        
        if mesh_path.endswith("textured.obj"):
            point_path = mesh_path.replace("textured.obj", "point_cloud.ply")
        else:
            point_path = mesh_path.replace("poly.obj", "point_cloud.ply")

        print(f"Processing mesh {i+1}/{len(mesh_files)}: {mesh_path}")
        print(f"Point cloud path: {point_path}")

        # Pass visualize=False since this is a batch script
        volume, psr_volumes, ransac_means, ransac_vars = compare_reconstruction_methods(mesh_path, point_path, n_trials=5, visualize=args.visualize)
        print(f"====== Object {i+1}/{len(mesh_files)} completed ======")   
        print(f"Volume of the mesh {mesh_path}: {volume}")

        psr_depths = [3,4,5,6,7]
        for psr_volume, depth in zip(psr_volumes, psr_depths):
            if psr_volume is not None:
                print(f"Volume from Poisson Surface Reconstruction (depth {depth}): {psr_volume}; Error {(abs(psr_volume-volume)/volume*100):.2f}%")
            else:
                print(f"Poisson reconstruction (depth {depth}) NON-WATERTIGHT.")

        print(f"Means from SphereRANSAC estimation: {ransac_means[0]}; Error {(abs(ransac_means[0]-volume)/volume*100):.2f}%")
        print(f"Variances from Sphere RANSAC estimation: {ransac_vars[0]}")
        print(f"Means from Cuboid RANSAC estimation: {ransac_means[1]}; Error {(abs(ransac_means[1]-volume)/volume*100):.2f}%")
        print(f"Variances from Cuboid RANSAC estimation: {ransac_vars[1]}")

        # --- Write results to CSV ---
        if args.store_results and csv_writer is not None:
            object_folder_name = Path(mesh_path).parent.name

            # --- FIX: Calculate errors and interleave data for the row ---
            psr_errors = [calculate_error_pct(v, volume) for v in psr_volumes]
            sphere_error = calculate_error_pct(ransac_means[0], volume)
            cuboid_error = calculate_error_pct(ransac_means[1], volume)

            # Interleave PSR volumes and their errors
            psr_data = [item for pair in zip(psr_volumes, psr_errors) for item in pair]

            row_data = [
                object_folder_name,
                volume,
                *psr_data, # Unpack the list of [vol3, err3, vol4, err4, ...]
                ransac_means[0], sphere_error, ransac_vars[0], # Sphere results
                ransac_means[1], cuboid_error, ransac_vars[1]  # Cuboid results
            ]
            csv_writer.writerow(row_data)
        # --- End Write results ---

    # --- Close CSV file ---
    if csv_file is not None:
        csv_file.close()
    # --- End Close CSV file ---

if __name__ == "__main__":
    args = parse_args()
    main()
