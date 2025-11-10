"""
This script estimates the height of an object from depth maps and masks.
Different from HeightEstimation.py, this version assumes we don't have access to camera extrinsic. 
Thus, we aim to represent the height functions in some self-defined coordinate system (x,z) on the ground plane.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image 
import os
import matplotlib.pyplot as plt 

def parse_args():
    ap = argparse.ArgumentParser(description="Estimate object height from depth and mask.")
    ap.add_argument("--depth", required=True, help="Path to folder containing depth .npy files.")
    ap.add_argument("--mask", "--mask_path", dest="mask", required=True,
                    help="Path to object mask (.png/.jpg black&white or .npy).")
    ap.add_argument("--K_json", required=True, help="Path to cameras.json.")
    ap.add_argument("--view_index", type=int, required=True, help="View index in cameras.json.")
    ap.add_argument("--reduce", type=str, default="median", choices=["median", "mean"], help="Aggregator for Z.")
    return ap.parse_args()

def load_camera_from_json(cameras_json: str, view_index: int):
    j = json.loads(Path(cameras_json).read_text())
    if "views" not in j:
        raise ValueError("cameras.json missing 'views' array")
    if view_index < 0 or view_index >= len(j["views"]):
        raise ValueError(f"view_index {view_index} out of range (0..{len(j['views'])-1})")
    K = np.array(j["views"][view_index]["K"], dtype=np.float32)
    W = int(j.get("width", 0))
    H = int(j.get("height", 0))
    return K, W, H

def load_mask_any(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    """Load .png/.jpg mask (white=object) or .npy; return boolean HxW. Resizes to target_hw if needed."""
    H, W = target_hw
    p = Path(path)
    if p.suffix.lower() == ".npy":
        m = np.load(p)
        if m.shape != (H, W):
            raise ValueError(f"Mask .npy shape {m.shape} != ({H},{W})")
        return m.astype(bool)
    # Image mask
    img = Image.open(p).convert("L")
    if img.size != (W, H):
        img = img.resize((W, H), Image.NEAREST)
    arr = np.array(img)
    return (arr > 127)  # white/object → True


def get_height_functions(depth: np.ndarray, K: np.ndarray, mask: np.ndarray, variable: str, footprint_length: float):
    """
    For the input image, return a height function (either h(z) or h(x)) based on viewing direction.
    Currently, we don't consider viewpoints at arbitrary angles (say 45 degrees), only axis-aligned views.

    variable: 'x', 'y', or 'z' indicating we want to get a function h(x) or h(z)
    - Note: if variable is x, we are viewing along z-axis
    - Note: if variable is z, we are viewing along x-axis

    footprint_length: physical length of the footprint in the corresponding direction (meters)
    - If variable is 'z', footprint_length is length in z direction; vise versa for 'x'

    Return:
        z/x-coordinates (1D array)
        corresponding heights (1D array)
    The indices across the two arrays align.
    """

    if variable not in ['x', 'z']:
        raise NotImplementedError("viewing_axis must be 'x' or 'z'")

    H, W = depth.shape

    # Get the span of z (valid coordinate)
    valid = mask & np.isfinite(depth) & (depth > 0)
    print(f"shape of valid: {valid.shape}, num valid pixels: {np.sum(valid)}")
    if variable == 'z':
        valid_pixels = np.where(valid)
        y_min = np.where(valid)[0].min()  # topmost - defines the lowest point in y-direction
        y_max = np.where(valid)[0].max()  # bottommost
        z_min = np.where(valid)[1].min()  # leftmost
        z_max = np.where(valid)[1].max()  # rightmost

        # Using the footprint length z to determine the scale factor
        scale_z = footprint_length / (z_max - z_min + 1)  # m/pixel

        valid_z_coordinates = []
        h_min_z = []
        h_max_z = []

        for z in range(z_min, z_max + 1):
            # Extract the maximum y and minimum y at this z
            column_valid = valid[:, z]
            if np.any(column_valid):
                ys = np.where(column_valid)[0]
                h_max = ys.max()
                h_min = ys.min()
                h_max_m = (h_max - y_min) * scale_z
                h_min_m = (h_min - y_min) * scale_z

                z_coordinate = (z - z_min) * scale_z  # convert into footprint coordinate
                valid_z_coordinates.append(z_coordinate)
                h_max_z.append(h_max_m)
                h_min_z.append(h_min_m)
        
        return np.array(valid_z_coordinates), np.array(h_max_z), np.array(h_min_z)

    if variable == 'x':
        valid_pixels = np.where(valid)
        y_min = np.where(valid)[0].min()  # topmost - defines the lowest point in y-direction
        y_max = np.where(valid)[0].max()  # bottommost
        x_min = np.where(valid)[1].min()  # leftmost
        x_max = np.where(valid)[1].max()  # rightmost

        # Using the footprint length x to determine the scale factor
        scale_x = footprint_length / (x_max - x_min + 1)  # m/pixel

        valid_x_coordinates = []
        h_min_x = []
        h_max_x = []

        for x in range(x_min, x_max + 1):
            # Extract the maximum y and minimum y at this x
            column_valid = valid[:, x]
            if np.any(column_valid):
                ys = np.where(column_valid)[0]
                h_max = ys.max()
                h_min = ys.min()
                h_max_m = (h_max - y_min) * scale_x
                h_min_m = (h_min - y_min) * scale_x

                x_coordinate = (x - x_min) * scale_x  # convert into footprint coordinate
                valid_x_coordinates.append(x_coordinate)
                h_max_x.append(h_max_m)
                h_min_x.append(h_min_m)

        return np.array(valid_x_coordinates), np.array(h_max_x), np.array(h_min_x)


def get_footprint(depth: np.ndarray, K: np.ndarray, mask: np.ndarray):
    """
    This applies to the top view only. We want to get the (x,z) footprint of the object to define the range of height function h(x,z).
    Return Nx2 array of (x,z) points in world coordinates, and an average pixel area (m^2).
    """
    H, W = depth.shape
    
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("No valid depth inside mask.")

    # Get the leftmost and bottommost points in pixel coordinates (valid pixels only)
    valid_pixels = np.where(valid)
    x_min = np.where(valid)[1].min()
    z_min = np.where(valid)[0].min()
    x_max = np.where(valid)[1].max()
    z_max = np.where(valid)[0].max()

    # Assume we use the center pixel's depth for scale factor calculation
    center_u = (x_min + x_max) // 2
    center_v = (z_min + z_max) // 2
    center_depth = depth[center_v, center_u]

    # Get the physical length of box in x and z directions, by using depth and intrinsics
    # NOTE: focal lengths are defined (W_px/2)/tan(fovx/2), so we can use them directly
    # See full derivation in Notes
    length_x = (x_max - x_min + 1) * center_depth / K[0, 0]
    length_z = (z_max - z_min + 1) * center_depth / K[1, 1]

    scale_x = length_x / (x_max - x_min + 1)
    scale_z = length_z / (z_max - z_min + 1)
    pixel_area = scale_x * scale_z  # m^2
    print(f"Footprint pixel bbox: u [{x_min}, {x_max}], v [{z_min}, {z_max}]")
    print(f"Footprint physical size: x {length_x:.6f} m, z {length_z:.6f} m")
    print(f"Footprint scale factors: x {scale_x:.6f} m/pixel, z {scale_z:.6f} m/pixel")

    # Convert the footprint pixels to world coordinates (x,z)
    valid_pixels = np.where(valid)
    footprint_x_z = []
    for v, u in zip(valid_pixels[0], valid_pixels[1]):
        d = depth[v, u]
        x = (u - x_min) * scale_x
        z = (v - z_min) * scale_z
        footprint_x_z.append([x, z])
    footprint_x_z = np.array(footprint_x_z, dtype=np.float32)

    print(f"Extracted {footprint_x_z.shape[0]} footprint points.")

    return footprint_x_z, pixel_area, length_x, length_z

def height_estimation(depth_folder: str, mask_folder: str, K_json: str):

    # Top View (assume index 0)
    print("=="*20)
    print(f"Processing Top View")
    depth_map_path = os.path.join(depth_folder, f"view_000_depth.npy")
    mask_path = os.path.join(mask_folder, f"resized_view_000_img_mask_1.png")

    depth = np.load(depth_map_path).astype(np.float32)
    H, W = depth.shape
    mask = load_mask_any(mask_path, (H, W))

    # Use top view index 0 for top view
    K_top, Wj, Hj = load_camera_from_json(K_json, 0)
    if Hj and Wj and (H != Hj or W != Wj):
        raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

    footprint_points, pixel_area, footprint_length_x, footprint_length_z = get_footprint(depth, K_top, mask)
    print(f"Footprint points shape {footprint_points.shape} extracted.")


    # Two side views: indices 1 and 2 (adjust if your cameras.json ordering differs)
    views = ["Side1", "Side2"]
    side_indices = [1, 2]

    for i, view in enumerate(views):
        print("=="*20)
        print(f"Processing view {i}: {view}")

        depth_map_path = os.path.join(depth_folder, f"view_00{i+1}_depth.npy")
        mask_path = os.path.join(mask_folder, f"resized_view_00{i+1}_img_mask_1.png")

        depth = np.load(depth_map_path).astype(np.float32)
        H, W = depth.shape
        mask = load_mask_any(mask_path, (H, W))

        K_i, Wj, Hj = load_camera_from_json(K_json, side_indices[i])
        if Hj and Wj and (H != Hj or W != Wj):
            raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

        if view == "Side1":
            z_coords_row, z_heights_max, z_heights_min = get_height_functions(depth, K_i, mask, "z", footprint_length_z)
            
            for i in range(len(z_coords_row)):
                print(f"z: {z_coords_row[i]:.6f} m, h_min: {z_heights_min[i]:.6f} m, h_max: {z_heights_max[i]:.6f} m")
        elif view == "Side2":
            x_coords_row, x_heights_max, x_heights_min = get_height_functions(depth, K_i, mask, "x", footprint_length_x)
            for i in range(len(x_coords_row)):
                print(f"x: {x_coords_row[i]:.6f} m, h_min: {x_heights_min[i]:.6f} m, h_max: {x_heights_max[i]:.6f} m")

    # Integration remains as you have; consider weighting by per-cell area instead of mean_area for better accuracy.
    height_total = 0.0
    x = []
    z = []
    h_max = []
    h_min = []
    h = []
    for i in range(footprint_points.shape[0]):
        x_pt, z_pt = footprint_points[i]
        # Nearest-neighbor in 1D (approx). For better accuracy, project (x_pt,z_pt,0) into each side image and sample there.
        x_idx = np.argmin(np.abs(x_coords_row - x_pt))
        z_idx = np.argmin(np.abs(z_coords_row - z_pt))
        height_max = min(x_heights_max[x_idx], z_heights_max[z_idx])
        height_min = max(x_heights_min[x_idx], z_heights_min[z_idx])
        height = height_max - height_min
        height_total += height

        x.append(x_pt)
        z.append(z_pt)
        h.append(height)
        h_max.append(height_max)
        h_min.append(height_min)

    volume_estimate = height_total * pixel_area
    print("=="*20)
    print(f"Estimated Volume of Object: {volume_estimate:.6f} m^3 = {volume_estimate*1000:.6f} L")

    # Plot height map over footprint
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, z, c=h, cmap='viridis', s=10)
    plt.colorbar(sc, label='Height (m)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Z Coordinate (m)')
    plt.title('Height Map Over Footprint')
    plt.grid()

    # Make X and Z same scale in 2D
    ax2d = plt.gca()
    ax2d.set_aspect('equal', adjustable='box')
    ax2d.set_xlim(np.min(x), np.max(x))
    ax2d.set_ylim(np.min(z), np.max(z))

    output_dir = os.path.join(Path(depth_folder).parent, "height_estimation_output")
    os.makedirs(output_dir, exist_ok=True)

    height_fig_path = os.path.join(output_dir, 'height.png')
    print(f"Saving height map to {height_fig_path}")
    plt.savefig(height_fig_path)
    plt.close()

    # 3D scatter of h_min and h_max over (x,z)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    x_arr = np.asarray(x)
    z_arr = np.asarray(z)
    h_top = np.asarray(h_max)
    h_bottom = np.asarray(h_min)

    s1 = ax.scatter(x_arr, z_arr, h_top, c=h_top, cmap='viridis', s=6, label='Height (m)')
    s2 = ax.scatter(x_arr, z_arr, h_bottom, c=h_bottom, cmap='viridis', s=6, label='Height (m)')

    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Z Coordinate (m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('3D Scatter: Height Over Footprint')
    ax.view_init(elev=35, azim=-60)
    ax.legend(loc='upper right')
    fig.colorbar(s1, ax=ax, shrink=0.6, label='h_max (m)')
    plt.tight_layout()

    # Make X and Z same scale in 3D
    x_min, x_max = x_arr.min(), x_arr.max()
    z_min, z_max = z_arr.min(), z_arr.max()
    x_span = max(x_max - x_min, 1e-9)
    z_span = max(z_max - z_min, 1e-9)
    h_min_all = min(h_bottom.min(), h_top.min())
    h_max_all = max(h_bottom.max(), h_top.max())
    h_span = max(h_max_all - h_min_all, 1e-9)

    ax.set_box_aspect((x_span, z_span, h_span))



    out3d = os.path.join(output_dir, 'height_3d_scatter.png')
    print(f"Saving 3D scatter to {out3d}")
    fig.savefig(out3d, dpi=300)
    plt.close(fig)

    return volume_estimate

def main():
    args = parse_args()

    height_estimation(args.depth, args.mask, args.K_json)

    # # Top View (assume index 0)
    # print("=="*20)
    # print(f"Processing Top View")
    # depth_map_path = os.path.join(args.depth, f"view_000_depth.npy")
    # mask_path = os.path.join(args.mask, f"resized_view_000_img_mask_1.png")

    # depth = np.load(depth_map_path).astype(np.float32)
    # H, W = depth.shape
    # mask = load_mask_any(mask_path, (H, W))

    # # Use top view index 0 for top view
    # K_top, Wj, Hj = load_camera_from_json(args.K_json, 0)
    # if Hj and Wj and (H != Hj or W != Wj):
    #     raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

    # footprint_points, pixel_area, footprint_length_x, footprint_length_z = get_footprint(depth, K_top, mask)
    # print(f"Footprint points shape {footprint_points.shape} extracted.")


    # # Two side views: indices 1 and 2 (adjust if your cameras.json ordering differs)
    # views = ["Side1", "Side2"]
    # side_indices = [1, 2]

    # for i, view in enumerate(views):
    #     print("=="*20)
    #     print(f"Processing view {i}: {view}")

    #     depth_map_path = os.path.join(args.depth, f"view_00{i+1}_depth.npy")
    #     mask_path = os.path.join(args.mask, f"resized_view_00{i+1}_img_mask_1.png")

    #     depth = np.load(depth_map_path).astype(np.float32)
    #     H, W = depth.shape
    #     mask = load_mask_any(mask_path, (H, W))

    #     K_i, Wj, Hj = load_camera_from_json(args.K_json, side_indices[i])
    #     if Hj and Wj and (H != Hj or W != Wj):
    #         raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

    #     if view == "Side1":
    #         z_coords_row, z_heights_max, z_heights_min = get_height_functions(depth, K_i, mask, "z", footprint_length_z, args)
            
    #         for i in range(len(z_coords_row)):
    #             print(f"z: {z_coords_row[i]:.6f} m, h_min: {z_heights_min[i]:.6f} m, h_max: {z_heights_max[i]:.6f} m")
    #     elif view == "Side2":
    #         x_coords_row, x_heights_max, x_heights_min = get_height_functions(depth, K_i, mask, "x", footprint_length_x, args)
    #         for i in range(len(x_coords_row)):
    #             print(f"x: {x_coords_row[i]:.6f} m, h_min: {x_heights_min[i]:.6f} m, h_max: {x_heights_max[i]:.6f} m")

    # # Integration remains as you have; consider weighting by per-cell area instead of mean_area for better accuracy.
    # height_total = 0.0
    # x = []
    # z = []
    # h_max = []
    # h_min = []
    # for i in range(footprint_points.shape[0]):
    #     x_pt, z_pt = footprint_points[i]
    #     # Nearest-neighbor in 1D (approx). For better accuracy, project (x_pt,z_pt,0) into each side image and sample there.
    #     x_idx = np.argmin(np.abs(x_coords_row - x_pt))
    #     z_idx = np.argmin(np.abs(z_coords_row - z_pt))
    #     height_max = min(x_heights_max[x_idx], z_heights_max[z_idx])
    #     height_min = max(x_heights_min[x_idx], z_heights_min[z_idx])
    #     height = height_max - height_min
    #     height_total += height

    #     x.append(x_pt)
    #     z.append(z_pt)
    #     h_max.append(height_max)
    #     h_min.append(height_min)

    # volume_estimate = height_total * pixel_area
    # print("=="*20)
    # print(f"Estimated Volume of Object: {volume_estimate:.6f} m^3 = {volume_estimate*1000:.6f} L")

    # # Plot height map over footprint
    # plt.figure(figsize=(8,6))
    # sc = plt.scatter(x, z, c=h_max, cmap='viridis', s=10)
    # plt.colorbar(sc, label='Max Height (m)')
    # plt.xlabel('X Coordinate (m)')
    # plt.ylabel('Z Coordinate (m)')
    # plt.title('Height Map Over Footprint')
    # plt.grid()
    # output_dir = Path(__file__).parent
    # print(f"Saving height map to {output_dir / 'height.png'}")
    # plt.savefig(output_dir / 'height.png')
    # plt.close()

    # # 3D plot of the top surface h_max over (x,z)
    # fig = plt.figure(figsize=(9,7))
    # ax = fig.add_subplot(111, projection='3d')
    # x_arr = np.asarray(x)
    # z_arr = np.asarray(z)
    # h_top = np.asarray(h_max)
    # h_bottom = np.asarray(h_min)

    # ax.plot_trisurf(x_arr, z_arr, h_top, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X Coordinate (m)')
    # ax.set_ylabel('Z Coordinate (m)')
    # ax.set_zlabel('Height (m)')
    # ax.set_title('3D Height Map Over Footprint')
    # ax.view_init(elev=35, azim=-60)
    # plt.tight_layout()
    # output_dir = Path(__file__).parent
    # out3d = output_dir / 'height_3d.png'
    # print(f"Saving 3D height map to {out3d}")
    # fig.savefig(out3d, dpi=300)
    # plt.close(fig)

if __name__ == "__main__":
    main()