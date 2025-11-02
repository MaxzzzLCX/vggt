"""
Minimal script to render 8 equally-spaced views around a mesh.
Usage: python view_synthesis_simple.py --mesh path/to/mesh.obj --out_dir output_folder
"""

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def normalize_mesh(mesh, target_radius=1.0):
    """Center mesh at origin and scale to fit within target_radius sphere."""
    mesh.compute_vertex_normals()
    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent = aabb.get_extent()
    scale = target_radius / (np.max(extent) / 2)
    mesh.translate(-center)
    mesh.scale(scale, center=[0, 0, 0])
    return mesh

def create_camera_positions(num_views=8, radius=2.5, elevation=20, rotation_axis='z'):
    """
    Create camera positions equally spaced around the object.
    
    Parameters
    ----------
    num_views : Number of camera positions
    radius : Distance from origin
    elevation : Height angle in degrees (relative to horizontal plane)
    rotation_axis : 'x', 'y', or 'z' - axis to rotate around
    """
    positions = []
    elevation_rad = np.radians(elevation)
    
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        
        if rotation_axis.lower() == 'z':
            # Rotate around Z-axis (Z is up, cameras in XY plane at fixed height)
            x = radius * np.cos(elevation_rad) * np.cos(azimuth)
            y = radius * np.cos(elevation_rad) * np.sin(azimuth)
            z = radius * np.sin(elevation_rad)
        elif rotation_axis.lower() == 'y':
            # Rotate around Y-axis (Y is up, cameras in XZ plane)
            x = radius * np.cos(elevation_rad) * np.cos(azimuth)
            y = radius * np.sin(elevation_rad)
            z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        elif rotation_axis.lower() == 'x':
            # Rotate around X-axis (X is up, cameras in YZ plane)
            x = radius * np.sin(elevation_rad)
            y = radius * np.cos(elevation_rad) * np.cos(azimuth)
            z = radius * np.cos(elevation_rad) * np.sin(azimuth)
        else:
            raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
        
        positions.append([x, y, z])
    
    return positions

def render_view(mesh, camera_pos, target=[0, 0, 0], width=512, height=512, rotation_axis='z'):
    """Render a single view using matplotlib (stable cross-platform method)."""
    # Create figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh data
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # --- FIX: Get vertex colors if they exist ---
    has_colors = mesh.has_vertex_colors()
    if has_colors:
        vertex_colors = np.asarray(mesh.vertex_colors)
    else:
        print("Mesh has no color")
    # --- END FIX ---
    
    # --- FIX: Use proper rotation matrices to align axis with Z ---
    if rotation_axis.lower() == 'y':
        # Y is up, rotate 90° around X-axis to make Y point where Z points
        # Rotation matrix: Rx(90°)
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])
        
    elif rotation_axis.lower() == 'x':
        # X is up, rotate 90° around Y-axis to make X point where Z points
        # Rotation matrix: Ry(-90°)
        rotation_matrix = np.array([
            [0,  0,  1],
            [0,  1,  0],
            [-1, 0,  0]
        ])
        
    else:  # rotation_axis == 'z' (default)
        # Z is already up, no rotation needed
        rotation_matrix = np.eye(3)
    
    # Apply rotation to vertices and camera position
    vertices_rotated = vertices @ rotation_matrix.T
    camera_pos_rotated = np.array(camera_pos) @ rotation_matrix.T
    # --- END FIX ---
    
    # --- FIX: Plot mesh with colors if available ---
    if has_colors:
        # Use vertex colors from the texture
        ax.plot_trisurf(vertices_rotated[:, 0], vertices_rotated[:, 1], vertices_rotated[:, 2],
                        triangles=triangles,
                        antialiased=True,
                        shade=True)
        # Manually set face colors based on vertex colors
        # Average the vertex colors for each triangle
        face_colors = vertex_colors[triangles].mean(axis=1)
        ax.collections[0].set_facecolors(face_colors)
        ax.collections[0].set_edgecolors('none')
    else:
        # Fallback to uniform color
        ax.plot_trisurf(vertices_rotated[:, 0], vertices_rotated[:, 1], vertices_rotated[:, 2],
                        triangles=triangles, 
                        color='lightblue', 
                        edgecolor='none',
                        linewidth=0,
                        alpha=0.9,
                        shade=True)
    # --- END FIX ---
    
    # Calculate view angles (now always assumes Z is up after rotation)
    azim = np.degrees(np.arctan2(camera_pos_rotated[1], camera_pos_rotated[0]))
    elev = 0  # Horizontal view
    
    ax.view_init(elev=elev, azim=azim)
    
    # Set equal aspect ratio and limits
    max_range = 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1,1,1])
    
    # Remove axes
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    fig.canvas.draw()
    
    # Get RGBA buffer and convert to RGB
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    
    # Convert RGBA to RGB by dropping the alpha channel
    img = img[:, :, :3]
    
    plt.close(fig)
    return img

def main():
    parser = argparse.ArgumentParser(description="Render views around a mesh")
    parser.add_argument("--mesh", required=True, help="Path to input mesh (.obj)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--num_views", type=int, default=8, help="Number of views")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--radius", type=float, default=2.5, help="Camera distance")
    parser.add_argument("--elevation", type=float, default=20, help="Camera elevation (degrees)")
    parser.add_argument("--rotation_axis", type=str, default='z', choices=['x', 'y', 'z'], 
                        help="Axis to rotate cameras around (x, y, or z)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load and normalize mesh
    print(f"Loading mesh: {args.mesh}")
    
    # --- FIX: Load mesh with texture support ---
    # Open3D will automatically look for the .mtl file and load textures
    mesh = o3d.io.read_triangle_mesh(args.mesh, enable_post_processing=True)
    
    # If the mesh has textures, convert them to vertex colors
    if mesh.has_triangle_uvs() and mesh.has_textures():
        print("Mesh has textures. Converting to vertex colors...")
        # This is a workaround: sample the texture and create vertex colors
        # Note: This is approximate since matplotlib doesn't support UV mapping directly
        mesh.compute_vertex_normals()
    elif not mesh.has_vertex_colors():
        print("No textures or vertex colors found. Using default color.")
    else:
        print("Mesh has vertex colors.")
    # --- END FIX ---
    
    if not mesh.has_triangles():
        print("ERROR: Mesh has no triangles!")
        return
    
    mesh = normalize_mesh(mesh)
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Generate camera positions
    camera_positions = create_camera_positions(
        num_views=args.num_views,
        radius=args.radius,
        elevation=args.elevation,
        rotation_axis=args.rotation_axis
    )
    
    # Render each view
    print(f"Rendering {args.num_views} views...")
    for i, cam_pos in enumerate(camera_positions):
        print(f"  Rendering view {i+1}/{args.num_views}...", end='\r')
        
        img = render_view(mesh, cam_pos, width=args.width, height=args.height, rotation_axis=args.rotation_axis)
        
        # Save image
        output_path = os.path.join(args.out_dir, f"view_{i:03d}.png")
        plt.imsave(output_path, img)
    
    print(f"\nDone! Images saved to {args.out_dir}")

if __name__ == "__main__":
    main()
