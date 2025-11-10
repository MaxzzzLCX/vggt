"""
Generates synthetic view images for all objects in a dataset folder
"""

import argparse
import os
from pathlib import Path
from orthogonal_view_synthesis import orthogonal_view_synthesis

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic views for dataset")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Directory containing dataset')
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--radius", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--elevation", type=float, default=0)
    parser.add_argument("--rotation_axis", type=str, default="y", choices=["x","y","z"])
    parser.add_argument("--yfov_deg", type=float, default=50.0)
    parser.add_argument("--start_index", type=int, required=True, help="Start index of objects to process")
    parser.add_argument("--end_index", type=int, required=True, help="End index of objects to process")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder

    total_folders = len(sorted(list(Path(dataset_folder).glob('id-*/')))[args.start_index:args.end_index])
    print(f"Processing {total_folders} objects from index {args.start_index} to {args.end_index} in {dataset_folder}")

    for i, object_folder in enumerate(sorted(list(Path(dataset_folder).glob('id-*/')))[args.start_index:args.end_index]):
        mesh_path = list(Path(object_folder).glob('*.obj'))[0] # The .obj file might have different names, either 'textured.obj' or 'poly.obj'
        out_dir = object_folder / 'images'
        os.makedirs(out_dir, exist_ok=True)

        # Call the view synthesis function (assuming it's defined elsewhere)
        orthogonal_view_synthesis(
            mesh_path=mesh_path, output_dir=out_dir, width=args.width, height=args.height,
            radius=args.radius, elevation=args.elevation, rotation_axis=args.rotation_axis,
            yfov_deg=args.yfov_deg
        )

        print(f"[{i+1}/{total_folders}] Processed {object_folder.name}")

if __name__ == "__main__":
    main()