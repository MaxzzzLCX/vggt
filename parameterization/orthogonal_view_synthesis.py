"""
Render textured meshes with proper UV mapping support.
Requires: pip install trimesh pyrender pillow
Usage: python view_synthesis_textured.py --mesh path/to/mesh.obj --out_dir output_folder
"""

import os
import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image
import json

def normalize_mesh(mesh, target_radius=1.0, return_scale=False):
    """Center mesh at origin and scale to fit within target_radius sphere."""
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = bounds.ptp(axis=0)
    scale = target_radius / (extent.max() / 2.0)
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    if return_scale:
        return mesh, scale
    return mesh

def create_camera_positions(num_views=8, radius=2.5, elevation=20, rotation_axis='y'):
    """Create camera positions equally spaced around the object."""
    positions = []
    elevation_rad = np.radians(elevation)
    
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        
        if rotation_axis.lower() == 'z':
            x = radius * np.cos(elevation_rad) * np.cos(azimuth)
            y = radius * np.cos(elevation_rad) * np.sin(azimuth)
            z = radius * np.sin(elevation_rad)
        elif rotation_axis.lower() == 'y':
            x = radius * np.cos(elevation_rad) * np.cos(azimuth)
            y = radius * np.sin(elevation_rad)
            z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        elif rotation_axis.lower() == 'x':
            x = radius * np.sin(elevation_rad)
            y = radius * np.cos(elevation_rad) * np.cos(azimuth)
            z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        
        positions.append([x, y, z])
    
    return positions


def create_camera_positions_orthogonal(radius=2.5, elevation=0, rotation_axis='y'):
    """Create camera positions in three orthogonal viewpoints: top, side, side."""
    positions = []
    elevation_rad = np.radians(elevation)
    
    # Top view
    positions.append([0, radius, 0])

    # Side view 1
    if rotation_axis.lower() == 'y':
         
        # Side view 1: azimuth = 0
        azimuth = 2 * np.pi * 0 
        x = radius * np.cos(elevation_rad) * np.cos(azimuth)
        y = radius * np.sin(elevation_rad)
        z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        positions.append([x, y, z])

        # Side view 2: azimuth = 90 degrees
        azimuth = np.pi * 0.5
        x = radius * np.cos(elevation_rad) * np.cos(azimuth)
        y = radius * np.sin(elevation_rad)
        z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        positions.append([x, y, z])

    else:
        raise NotImplementedError("Currently only 'y' rotation_axis is supported for orthogonal views.")
    
    return positions


def camera_intrinsics_from_yfov(width, height, yfov_deg):
    """Compute K (square pixels, zero skew) from vertical FOV and image size."""
    yfov = np.radians(yfov_deg)
    fy = (height / 2.0) / np.tan(yfov / 2.0)
    # derive horizontal FOV from aspect ratio
    fovx = 2.0 * np.arctan((width / height) * np.tan(yfov / 2.0))
    fx = (width / 2.0) / np.tan(fovx / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def render_view(mesh, camera_pos, target=[0, 0, 0], width=518, height=518, yfov_deg=50.0):
    """Returns color (HxWx3 uint8), depth (HxW float32 in scene units), K (3x3), T_wc (4x4)."""
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=np.eye(4))

    K = camera_intrinsics_from_yfov(width, height, yfov_deg)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(yfov_deg))

    # Build T_wc (camera-to-world).
    # pyrender expects the camera to look along its -Z axis, so set +Z to point from target -> camera (backward).
    z = np.array(camera_pos, dtype=np.float32) - np.array(target, dtype=np.float32)  # camera +Z = backward
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T_wc = np.eye(4, dtype=np.float32)
    T_wc[:3, 0] = x
    T_wc[:3, 1] = y
    T_wc[:3, 2] = z
    T_wc[:3, 3] = np.array(camera_pos, dtype=np.float32)

    scene.add(camera, pose=T_wc)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth_scene = renderer.render(scene)  # depth is Z-depth in scene units
    renderer.delete()
    return color, depth_scene, K, T_wc

def orthogonal_view_synthesis(mesh_path, output_dir, width=518, height=518, 
                              radius=2.5, elevation=0, rotation_axis='y', yfov_deg=50.0,):
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Normalize for stable framing and get scale
    mesh, scale = normalize_mesh(mesh, return_scale=True)

    print(f"Mesh faces: {len(mesh.faces)}  vertices: {len(mesh.vertices)}")

    camera_positions = create_camera_positions_orthogonal(
        radius=radius,
        elevation=elevation,
        rotation_axis=rotation_axis
    )

    meta = {
        "width": width,
        "height": height,
        "yfov_deg": yfov_deg,
        "views": []
    }
    Ks, T_wcs = [], []

    for i, cam_pos in enumerate(camera_positions):
        print(f"  Rendering view {i+1}/{3}...", end="\r")
        color, depth_norm, K, T_wc = render_view(
            mesh, cam_pos, width=width, height=height, yfov_deg=yfov_deg
        )

        # Convert Z-depth to metric assuming original mesh units were meters
        depth_m = depth_norm.astype(np.float32) / max(scale, 1e-12)

        img_path = os.path.join(output_dir, f"view_{i:03d}_img.png")
        depth_path = os.path.join(output_dir, f"view_{i:03d}_depth.npy")
        Image.fromarray(color).save(img_path)
        np.save(depth_path, depth_m)

        # Make pose translation metric to match depth units
        T_wc_metric = T_wc.copy()
        T_wc_metric[:3, 3] /= max(scale, 1e-12)

        meta["views"].append({
            "name": f"view_{i:03d}",
            "K": K.tolist(),
            "T_wc": T_wc_metric.tolist(),  # metric pose
            "img": os.path.basename(img_path),
            "depth_npy": os.path.basename(depth_path)
        })
        Ks.append(K)
        T_wcs.append(T_wc_metric)

    with open(os.path.join(output_dir, "cameras.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: save calib for precise loading
    np.savez_compressed(
        os.path.join(output_dir, "calib.npz"),
        K=np.stack(Ks, 0).astype(np.float32),
        T_wc=np.stack(T_wcs, 0).astype(np.float32),
        width=np.int32(width),
        height=np.int32(height),
        yfov_deg=np.float32(yfov_deg),
        scale=np.float32(scale),
    )
    print("\nDone. Saved RGB, depth (meters via 1/scale), and calibration.")

def main():
    parser = argparse.ArgumentParser(description="Render textured views around a mesh")
    parser.add_argument("--mesh", required=True, help="Path to input mesh (.obj)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--radius", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--elevation", type=float, default=20)
    parser.add_argument("--rotation_axis", type=str, default="y", choices=["x","y","z"])
    parser.add_argument("--yfov_deg", type=float, default=50.0)
    args = parser.parse_args()

    orthogonal_view_synthesis(
        mesh_path = args.mesh,
        out_dir = args.out_dir,
        width = args.width,
        height = args.height,
        radius = args.radius,
        elevation = args.elevation,
        rotation_axis = args.rotation_axis,
        yfov_deg = args.yfov_deg,
    )

if __name__ == "__main__":
    main()