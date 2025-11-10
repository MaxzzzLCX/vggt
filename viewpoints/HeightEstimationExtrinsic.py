import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image  # NEW
import os
from matplotlib import pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Estimate object height from depth and mask.")
    ap.add_argument("--depth", required=True, help="Path to folder containing depth .npy files.")
    ap.add_argument("--mask", "--mask_path", dest="mask", required=True,
                    help="Path to object mask (.png/.jpg black&white or .npy).")
    ap.add_argument("--K_json", required=True, help="Path to cameras.json.")
    ap.add_argument("--reduce", type=str, default="median", choices=["median", "mean"], help="Aggregator for Z.")
    return ap.parse_args()

def load_camera_from_json(cameras_json: str, view_index: int):
    j = json.loads(Path(cameras_json).read_text())
    if "views" not in j:
        raise ValueError("cameras.json missing 'views' array")
    if view_index < 0 or view_index >= len(j["views"]):
        raise ValueError(f"view_index {view_index} out of range (0..{len(j['views'])-1})")
    K = np.array(j["views"][view_index]["K"], dtype=np.float32)
    T_wc = np.array(j["views"][view_index]["T_wc"], dtype=np.float32)
    W = int(j.get("width", 0))
    H = int(j.get("height", 0))
    return K, T_wc, W, H

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

def get_height_in_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Given boolean HxW mask
    Return a 1xW array of object height in pixels at each columnn
    """
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    for x in range(W):
        ys = np.where(mask[:, x])[0]
        if len(ys) > 0:
            heights[x] = ys.max() - ys.min() + 1
    return heights

def heights_from_side_world(world:np.ndarray, depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-column height in meters from a side view by unprojecting to world and taking Y-range.
    Returns:
      u_idx: (W,) column indices
      heights_min: (W,) height per column (NaN where no valid mask)
      heights_max: (W,) height per column (NaN where no valid mask)
    """
    H, W = depth.shape
    y = world[..., 1]  # world vertical (meters)
    valid = mask & np.isfinite(depth) & (depth > 0) & np.isfinite(y)

    # Set invalid to NaN, then reduce along rows (v)
    y_masked = np.where(valid, y, np.nan).astype(np.float32)
    y_min = np.nanmin(y_masked, axis=0)  # (W,)
    y_max = np.nanmax(y_masked, axis=0)  # (W,)
    heights_min = y_min
    heights_max = y_max
    heights = y_max - y_min              # (W,)

    # If a column has no valid pixels, set height to NaN
    no_valid_col = ~np.any(valid, axis=0)
    heights_min[no_valid_col] = np.nan
    heights_max[no_valid_col] = np.nan
    heights[no_valid_col] = np.nan
    u_idx = np.arange(W, dtype=np.int32)
    return u_idx, heights_min, heights_max

def get_height_functions(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray, variable: str):
    """
    For the input image, return a height function (either h(z) or h(x)) based on viewing direction.
    Currently, we don't consider viewpoints at arbitrary angles (say 45 degrees), only axis-aligned views.

    variable: 'x', 'y', or 'z' indicating we want to get a function h(x) or h(z)
    - Note: if variable is x, we are viewing along z-axis
    - Note: if variable is z, we are viewing along x-axis

    Return:
        z/x-coordinates (1D array)
        corresponding heights (1D array)
    The indices across the two arrays align.
    """

    if variable not in ['x', 'z']:
        raise NotImplementedError("viewing_axis must be 'x' or 'z'")

    H, W = depth.shape

    
    # World coords map for mapping column u to world x/z at center row (approximate)
    world_coords = unproject_mask_to_world(depth, K, T_wc, mask=None)

    # Compute per-column heights from 3D (world Y range), robust to perspective
    u_idx, height_min_m, height_max_m = heights_from_side_world(world_coords,depth, K, T_wc, mask)

    center_v = H // 2
    center_row_world = world_coords[center_v, :, :]  # Wx3
    valid_row = np.isfinite(center_row_world[:, 0])

    if variable == 'z':
        z_coords_row = center_row_world[valid_row, 2]
        z_heights_min = height_min_m[valid_row]
        z_heights_max = height_max_m[valid_row]
        # Optionally drop NaNs
        m = np.isfinite(z_heights_min) & np.isfinite(z_heights_max)
        z_coords_row, z_heights_min, z_heights_max = z_coords_row[m], z_heights_min[m], z_heights_max[m]
        return z_coords_row, z_heights_min, z_heights_max

    elif variable == 'x':
        x_coords_row = center_row_world[valid_row, 0]
        x_heights_min = height_min_m[valid_row]
        x_heights_max = height_max_m[valid_row]
        m = np.isfinite(x_heights_min) & np.isfinite(x_heights_max)
        x_coords_row, x_heights_min, x_heights_max = x_coords_row[m], x_heights_min[m], x_heights_max[m]
        return x_coords_row, x_heights_min, x_heights_max

def pixel_area_xz_from_world(world_coords: np.ndarray, mask: np.ndarray | None = None):
    """
    Compute per-cell area in the x–z plane via a 2x2 Jacobian determinant, vectorized.
    Returns:
      area_map: (H-1, W-1) array of areas (m^2) for each pixel cell (top-left anchor)
      cell_valid: (H-1, W-1) bool where all four corners are valid (and in mask if provided)
      mean_area: scalar mean over valid cells (np.nan if none)
    """
    x = world_coords[..., 0]
    z = world_coords[..., 2]

    finite = np.isfinite(x) & np.isfinite(z)
    if mask is not None:
        finite &= mask.astype(bool)

    # Differences along image axes (u → columns, v → rows)
    # Build cell edges using the top-left corner as anchor.
    dx_du = x[:-1, 1:] - x[:-1, :-1]  # along +u at top row of each cell
    dz_du = z[:-1, 1:] - z[:-1, :-1]
    dx_dv = x[1:, :-1] - x[:-1, :-1]  # along +v at left col of each cell
    dz_dv = z[1:, :-1] - z[:-1, :-1]

    # 2x2 Jacobian determinant for (x,z) w.r.t (u,v): |dx/du  dx/dv; dz/du  dz/dv|
    area_map = np.abs(dx_du * dz_dv - dz_du * dx_dv).astype(np.float32)

    # Cell is valid if all four corners are valid
    v00 = finite[:-1, :-1]
    v01 = finite[:-1, 1:]
    v10 = finite[1:, :-1]
    v11 = finite[1:, 1:]
    cell_valid = v00 & v01 & v10 & v11

    mean_area = float(area_map[cell_valid].mean()) if np.any(cell_valid) else np.nan
    return area_map, cell_valid, mean_area

def get_footprint(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray):
    """
    This applies to the top view only. We want to get the (x,z) footprint of the object to define the range of height function h(x,z).
    Return Nx2 array of (x,z) points in world coordinates, and an average pixel area (m^2).
    """
    H, W = depth.shape
    
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("No valid depth inside mask.")

    # Calculate world points from mask
    world_coords = unproject_mask_to_world(depth, K, T_wc, mask=mask)
    print(f"World coords map shape: {world_coords.shape}")

    # Vectorized average pixel area in x–z plane
    area_map, cell_valid, mean_area = pixel_area_xz_from_world(world_coords, mask=mask)
    num_cells = int(cell_valid.sum())
    print(f"Valid footprint cells: {num_cells} / {(H-1)*(W-1)}")
    print(f"Average pixel area in world x–z plane: {mean_area} m^2")

    # Flatten valid footprint points (x,z)
    footprint_points = world_coords[valid]
    footprint_x_z = footprint_points[:, [0, 2]]  # Nx2 array of (x,z)

    return footprint_x_z, mean_area


def unproject_mask_to_world(depth_m, K, T_wc, mask=None):
    """
    Return dense world coordinates for each pixel: (H, W, 3), NaN where invalid/masked.
    depth_m: HxW Z-depth (meters), K: 3x3, T_wc: 4x4 camera->world (metric).
    """
    H, W = depth_m.shape
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    # Validity
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= mask.astype(bool)
    # Pixel grids
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    Z = depth_m.astype(np.float32)
    # Camera coordinates
    Xc_x = (u - cx) / fx * Z
    Xc_y = (v - cy) / fy * Z
    Xc_z = Z
    # World coordinates: Xw = R * Xc + t
    R = T_wc[:3, :3].astype(np.float32)
    t = T_wc[:3, 3].astype(np.float32)
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    Xw_x = r00 * Xc_x + r01 * Xc_y + r02 * Xc_z + t[0]
    Xw_y = r10 * Xc_x + r11 * Xc_y + r12 * Xc_z + t[1]
    Xw_z = r20 * Xc_x + r21 * Xc_y + r22 * Xc_z + t[2]
    # Invalidate non-valid pixels
    inv = ~valid
    Xw_x[inv] = np.nan
    Xw_y[inv] = np.nan
    Xw_z[inv] = np.nan
    # Stack to (H, W, 3)
    world = np.stack([Xw_x, Xw_y, Xw_z], axis=-1).astype(np.float32)
    return world

def height_estimation(depth_folder: str, mask_folder: str, K_json: str) -> float:

    # Top View (assume index 0)
    print("=="*20)
    print(f"Processing Top View")
    depth_map_path = os.path.join(depth_folder, f"view_000_depth.npy")
    mask_path = os.path.join(mask_folder, f"resized_view_000_img_mask_1.png")

    depth = np.load(depth_map_path).astype(np.float32)
    H, W = depth.shape
    mask = load_mask_any(mask_path, (H, W))

    # Use top view index 0 for top view
    K_top, T_wc_top, Wj, Hj = load_camera_from_json(K_json, 0)
    if Hj and Wj and (H != Hj or W != Wj):
        raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

    footprint_points, pixel_area = get_footprint(depth, K_top, T_wc_top, mask)
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

        K_i, T_wc_i, Wj, Hj = load_camera_from_json(K_json, side_indices[i])
        if Hj and Wj and (H != Hj or W != Wj):
            raise ValueError(f"Depth/mask shape ({H},{W}) != cameras.json ({Hj},{Wj})")

        if view == "Side1":
            z_coords_row, z_heights_min, z_heights_max = get_height_functions(depth, K_i, T_wc_i, mask, "z")
        elif view == "Side2":
            x_coords_row, x_heights_min, x_heights_max = get_height_functions(depth, K_i, T_wc_i, mask, "x")

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
        height_final = max(0.0, height_max - height_min)
        height_total += height_final

        x.append(x_pt)
        z.append(z_pt)
        h_max.append(height_max)
        h_min.append(height_min)
        h.append(height_final)


    volume_estimate = height_total * pixel_area
    print("=="*20)
    print(f"Estimated Volume of Object: {volume_estimate:.6f} m^3 = {volume_estimate*1000:.6f} L")

    ### PLOTTING ###
    # Plot height map over footprint
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, z, c=h, cmap='viridis', s=10)
    plt.colorbar(sc, label='Height (m)')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Z Coordinate (m)')
    plt.title('Height Map Over Footprint (Extrinsic)')
    plt.grid()

    # Make X and Z same scale in 2D
    ax2d = plt.gca()
    ax2d.set_aspect('equal', adjustable='box')
    ax2d.set_xlim(np.min(x), np.max(x))
    ax2d.set_ylim(np.min(z), np.max(z))

    output_dir = os.path.join(Path(depth_folder).parent, "height_estimation_extrinsic")
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
    ax.set_title('3D Scatter: Height Over Footprint (Extrinsic)')
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

    height_estimation(
        depth_folder=args.depth,
        mask_folder=args.mask,
        K_json=args.K_json
    )


if __name__ == "__main__":
    main()