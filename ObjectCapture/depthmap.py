"""
This file contains helper functions for saving previous depth maps from .npy to 16-bit .png format.
"""
import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

def save_as_png_16bit(depth_npy_path: str, output_png_path: str, scale: float = 1000.0, invalid_value: int = 0):
    """
    Save depth (meters, float32) as 16-bit PNG with quantization.
    - scale=1000 → store millimeters, precision ~1 mm, max range ~65.535 m.
    - invalid pixels (NaN, inf, <=0) are set to `invalid_value` (default 0).
    """
    depth_m = np.load(depth_npy_path).astype(np.float32)
    H, W = depth_m.shape
    print(f"[INFO] Loaded depth: shape={depth_m.shape}, min={np.nanmin(depth_m):.6f}, max={np.nanmax(depth_m):.6f}")

    invalid = ~np.isfinite(depth_m) | (depth_m <= 0)
    # Quantize to uint16 safely, clipping to [0, 65535]
    q = np.rint(depth_m * scale).astype(np.int64)
    clipped_high = (q > 65535).sum()
    q = np.clip(q, 0, 65535)

    q = q.astype(np.uint16)
    q[invalid] = np.uint16(invalid_value)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    Image.fromarray(q, mode='I;16').save(output_png_path)
    print(f"[INFO] Saved 16-bit depth PNG to: {output_png_path}")
    if clipped_high > 0:
        max_depth_m = 65535.0 / scale
        print(f"[WARN] {clipped_high} pixels clipped at max ({max_depth_m:.3f} m). Consider a smaller scale (e.g., 100).")

    # Save metadata sidecar for round-trip
    meta = {
        "width": int(W),
        "height": int(H),
        "scale_units_per_meter": float(scale),   # depth_units = depth_m * scale
        "unit_name": "millimeter" if scale == 1000.0 else "scaled_unit",
        "invalid_value": int(invalid_value),
        "stored_dtype": "uint16"
    }
    meta_path = os.path.splitext(output_png_path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Saved metadata JSON to: {meta_path}")

def load_depth_from_png_16bit(png_path: str, json_path: str | None = None) -> np.ndarray:
    """
    Load 16-bit PNG depth back to meters using metadata (preferred) or default scale=1000.
    """
    q = np.array(Image.open(png_path), dtype=np.uint16)
    scale = 1000.0
    invalid_value = 0
    if json_path is None:
        jp = os.path.splitext(png_path)[0] + ".json"
        if os.path.exists(jp):
            json_path = jp
    if json_path and os.path.exists(json_path):
        meta = json.load(open(json_path, "r"))
        scale = float(meta.get("scale_units_per_meter", scale))
        invalid_value = int(meta.get("invalid_value", invalid_value))
    depth_m = q.astype(np.float32) / scale
    depth_m[q == invalid_value] = np.nan
    return depth_m

def process_dataset(dataset_folder: str):
    """
    Process all depth .npy files in the dataset folder, converting them to 16-bit PNGs.
    Assumes depth .npy files are located in subfolders named 'images' under each object folder.
    """

    dataset_path = Path(dataset_folder)
    object_folders = sorted(list(dataset_path.glob('id-*')))
    total = len(object_folders)
    print(f"Total objects found: {total}")

    for i, object_folder in enumerate(object_folders):
        print(f"=="*20)
        print(f"[{i+1}/{total}] Processing object: {object_folder.name}")
        depth_folder = object_folder / 'images'
        output_depth_folder = object_folder / 'depth'
        os.makedirs(output_depth_folder, exist_ok=True)

        depth_npy_files = list(depth_folder.glob('*_depth.npy'))
        for depth_npy_path in depth_npy_files:
            output_png_path = output_depth_folder / (depth_npy_path.stem.replace('_depth', '_depth_mm') + '.png')
            save_as_png_16bit(
                depth_npy_path=str(depth_npy_path),
                output_png_path=str(output_png_path),
                scale=1000.0,
                invalid_value=0
            )
def main():
    # depth_npy_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_apple/images/view_000_depth.npy"
    # output_png_path = os.path.join(os.path.dirname(os.path.dirname(depth_npy_path)), "depth", "view_000_depth_mm.png")
    # save_as_png_16bit(depth_npy_path, output_png_path, scale=10000.0, invalid_value=0)
    
    dataset_folder = "/Users/maxlyu/Documents/nutritionverse-3d-dataset-manual"
    process_dataset(dataset_folder)

if __name__ == "__main__":
    main()