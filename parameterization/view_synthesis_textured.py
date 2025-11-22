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

def render_view(scene, camera_node, renderer, camera_pos, target=[0, 0, 0], width=518, height=518, yfov_deg=50.0):
    """Update camera pose and render. Returns color, depth, K, T_wc."""
    K = camera_intrinsics_from_yfov(width, height, yfov_deg)

    # Build T_wc (camera-to-world).
    z = np.array(camera_pos, dtype=np.float32) - np.array(target, dtype=np.float32)
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

    # Update camera pose in scene
    scene.set_pose(camera_node, T_wc)

    color, depth_scene = renderer.render(scene)
    return color, depth_scene, K, T_wc

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

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Normalize for stable framing and get scale
    mesh, scale = normalize_mesh(mesh, return_scale=True)

    print(f"Mesh faces: {len(mesh.faces)}  vertices: {len(mesh.vertices)}")

    # Build scene once
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=np.eye(4))
    camera = pyrender.PerspectiveCamera(yfov=np.radians(args.yfov_deg))
    camera_node = scene.add(camera, pose=np.eye(4))  # Placeholder pose

    # Create renderer once
    renderer = pyrender.OffscreenRenderer(args.width, args.height)

    camera_positions = create_camera_positions(
        num_views=args.num_views,
        radius=args.radius,
        elevation=args.elevation,
        rotation_axis=args.rotation_axis
    )

    meta = {
        "width": args.width,
        "height": args.height,
        "yfov_deg": args.yfov_deg,
        "views": []
    }
    Ks, T_wcs = [], []

    print(f"Rendering {args.num_views} views...")
    for i, cam_pos in enumerate(camera_positions):
        print(f"  Rendering view {i+1}/{args.num_views}...", end="\r")
        color, depth_norm, K, T_wc = render_view(
            scene, camera_node, renderer, cam_pos,
            width=args.width, height=args.height, yfov_deg=args.yfov_deg
        )

        # Convert Z-depth to metric assuming original mesh units were meters
        depth_m = depth_norm.astype(np.float32) / max(scale, 1e-12)

        img_path = os.path.join(args.out_dir, f"view_{i:03d}_img.png")
        depth_path = os.path.join(args.out_dir, f"view_{i:03d}_depth.npy")
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

    # Clean up renderer once at the end
    renderer.delete()

    with open(os.path.join(args.out_dir, "cameras.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, "calib.npz"),
        K=np.stack(Ks, 0).astype(np.float32),
        T_wc=np.stack(T_wcs, 0).astype(np.float32),
        width=np.int32(args.width),
        height=np.int32(args.height),
        yfov_deg=np.float32(args.yfov_deg),
        scale=np.float32(scale),
    )
    print("\nDone. Saved RGB, depth (meters via 1/scale), and calibration.")

if __name__ == "__main__":
    main()