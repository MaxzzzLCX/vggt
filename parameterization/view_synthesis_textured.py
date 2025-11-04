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

def render_view(mesh, camera_pos, target=[0, 0, 0], width=518, height=518):
    """
    Render a single view with proper texture support using pyrender.

    Width and height are 518 to match dimension required by VGGT.
    """
    # Create pyrender scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    
    # Add mesh to scene
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh))
    
    # Add directional light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))
    
    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.radians(50))
    camera_pose = np.eye(4)
    
    # Look at target from camera position
    z = np.array(camera_pos) - np.array(target)
    z = z / np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    camera_pose[:3, 0] = x
    camera_pose[:3, 1] = y
    camera_pose[:3, 2] = z
    camera_pose[:3, 3] = camera_pos
    
    scene.add(camera, pose=camera_pose)
    
    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()
    
    return color, depth

def main():
    parser = argparse.ArgumentParser(description="Render textured views around a mesh")
    parser.add_argument("--mesh", required=True, help="Path to input mesh (.obj)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views")
    parser.add_argument("--width", type=int, default=518, help="Image width")
    parser.add_argument("--height", type=int, default=518, help="Image height")
    parser.add_argument("--radius", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--elevation", type=float, default=20, help="Camera elevation (degrees)")
    parser.add_argument("--rotation_axis", type=str, default='y', choices=['x', 'y', 'z'])
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading mesh: {args.mesh}")
    # Trimesh automatically loads .mtl and textures
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene with multiple meshes, merge them
        mesh = trimesh.util.concatenate(mesh.dump())
    
    mesh, scale = normalize_mesh(mesh, return_scale=True)
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    if hasattr(mesh.visual, 'material'):
        print("✓ Mesh has texture/material")
    
    camera_positions = create_camera_positions(
        num_views=args.num_views,
        radius=args.radius,
        elevation=args.elevation,
        rotation_axis=args.rotation_axis
    )
    
    print(f"Rendering {args.num_views} views...")
    for i, cam_pos in enumerate(camera_positions):
        print(f"  Rendering view {i+1}/{args.num_views}...", end='\r')
        
        color, depth_norm = render_view(mesh, cam_pos, width=args.width, height=args.height)
        depth_m = depth_norm / scale  # if original mesh units were meters

        color_path = os.path.join(args.out_dir, f"view_{i:03d}_img.png")
        depth_path = os.path.join(args.out_dir, f"view_{i:03d}_depth.npy")

        Image.fromarray(color).save(color_path)
        np.save(depth_path, depth_m)

    print(f"\nDone! Images saved to {args.out_dir}")

if __name__ == "__main__":
    main()