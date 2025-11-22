"""
Autuomate the pipeline of volume estimation using the HeightEstimationExtrinsic.py script.
"""
import argparse
import os
from pathlib import Path
import open3d as o3d
import csv
import numpy as np

import HeightEstimationExtrinsic

def parse_args():
    parser = argparse.ArgumentParser(description="Test all volume estimation on a dataset.")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Directory containing dataset.')
    parser.add_argument("--start_index", type=int, required=True, help="Start index of objects to process")
    parser.add_argument("--end_index", type=int, required=True, help="End index of objects to process")
    parser.add_argument("--save_csv", action='store_true', help="Whether to save results to CSV file")
    parser.add_argument("--save_npy", action='store_true', help="Whether to save intermediate numpy files")
    parser.add_argument("--height_method", type=str, default="nearest", 
                    choices=["nearest", "interp1d", "project"],
                    help="Method to sample height at footprint: nearest=NN lookup, interp1d=linear interp, project=reproject to image")
    return parser.parse_args()



def fix_and_get_volume(path):
    """
    Load a mesh .obj file, fix its non-watertight issue, and return its volume.
    """

    mesh = o3d.io.read_triangle_mesh(path)

    # Remove texture coordinates if present
    if mesh.has_triangle_uvs():
            mesh.triangle_uvs = o3d.utility.Vector2dVector([])

    # Fix non-watertight issues
    if not mesh.is_watertight():
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    try:
        volume = mesh.get_volume()
        return volume
    except Exception as e:
        print(f"Error computing volume for {path}: {e}")
        return None

def main():
    args = parse_args()

    if args.save_csv:
        # Write CSV header
        result_csv = os.path.join(args.dataset_folder, f'height_estimation_extrinsic_M-{args.height_method}_{args.start_index}_{args.end_index}.csv')
        with open(result_csv, mode='w', newline='') as file:
            fieldnames = ['Object', 'GT Volume', 'Estimated Volume', 'Error Percentage']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    

    task_folder_length = len(sorted(list(Path(args.dataset_folder).glob('id-*/')))[args.start_index:args.end_index])

    all_objects = []
    all_gt_volumes = []
    all_estimated_volumes = []
    all_error_pcts = []


    for i, object_folder in enumerate(sorted(list(Path(args.dataset_folder).glob('id-*/')))[args.start_index:args.end_index]):
        print(f"**"*50)
        print(f"[{i+1}/{task_folder_length}] Processing {object_folder.name}...")
        print(f"**"*50)
        mesh_path = list(Path(object_folder).glob('*.obj'))[0] # The .obj file might have different names, either 'textured.obj' or 'poly.obj'
        gt_volume = fix_and_get_volume(mesh_path)

        depth_folder = object_folder / 'images'
        mask_folder = object_folder / 'masks'
        K_json_path = object_folder / 'images' / 'cameras.json'

        estimated_volume = HeightEstimationExtrinsic.height_estimation(
            depth_folder=str(depth_folder),
            mask_folder=str(mask_folder),
            K_json=str(K_json_path),
            height_method=args.height_method
        )

        if gt_volume is None:
            print(f"Estimated Volume: {estimated_volume}, GT Volume: N/A")
            print(f"Error Percentage: N/A")
            error_pct = None
        else:
            error_pct = abs(estimated_volume - gt_volume) / gt_volume * 100
            print(f"Estimated Volume: {estimated_volume}, GT Volume: {gt_volume}")
            print(f"Error Percentage: {error_pct:.4f}%")
        
        if args.save_csv:
            # Write results to CSV
            with open(result_csv, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({
                    'Object': object_folder.name,
                    'GT Volume': gt_volume if gt_volume is not None else 'N/A',
                    'Estimated Volume': estimated_volume,
                    'Error Percentage': f"{error_pct:.4f}%" if error_pct is not None else 'N/A'
                })
    
        # Save all results to .npy file
        if args.save_npy:
            all_objects.append(object_folder.name)
            all_gt_volumes.append(gt_volume)
            all_estimated_volumes.append(estimated_volume)
            all_error_pcts.append(error_pct)
    
    if args.save_npy:
        npz_path = os.path.join(args.dataset_folder, f'height_estimation_extrinsic_M-{args.height_method}_{args.start_index}_{args.end_index}.npz')
        np.savez_compressed(
            npz_path,
            objects=all_objects,
            gt_volumes=all_gt_volumes,
            estimated_volumes=all_estimated_volumes,
            error_percentages=all_error_pcts
        )
        print(f"Saved results to {npz_path}")

    
    # Check loading the npz
    data = np.load(npz_path, allow_pickle=True)
    print("Loaded data from npz:")
    print("Objects:", data['objects'])
    print("GT Volumes:", data['gt_volumes'])
    print("Estimated Volumes:", data['estimated_volumes'])
    print("Error Percentages:", data['error_percentages'])


if __name__ == "__main__":
    main()