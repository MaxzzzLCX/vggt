"""
Visualize a mesh with coordinate axes to understand its orientation.
Usage: python visualize_mesh_axes.py --mesh path/to/mesh.obj
"""

import argparse
import numpy as np
import open3d as o3d

def create_coordinate_frame(size=1.0, origin=[0, 0, 0]):
    """Create a coordinate frame with RGB axes (X=red, Y=green, Z=blue)."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin
    )

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

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh with coordinate axes")
    parser.add_argument("--mesh", required=True, help="Path to input mesh (.obj)")
    parser.add_argument("--axis_size", type=float, default=1.5, help="Size of coordinate axes")
    args = parser.parse_args()
    
    # Load mesh
    print(f"Loading mesh: {args.mesh}")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    
    if not mesh.has_triangles():
        print("ERROR: Mesh has no triangles!")
        return
    
    # Normalize mesh
    mesh = normalize_mesh(mesh, target_radius=1.0)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
    
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Create coordinate frame
    # X-axis = RED, Y-axis = GREEN, Z-axis = BLUE
    axes = create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])
    
    print("\n" + "="*60)
    print("COORDINATE SYSTEM:")
    print("  X-axis = RED   (right)")
    print("  Y-axis = GREEN (forward)")
    print("  Z-axis = BLUE  (up)")
    print("="*60)
    print("\nInteractive controls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'H' for help")
    print("="*60 + "\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [mesh, axes],
        window_name="Mesh with Coordinate Axes",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()