import open3d as o3d
import argparse
import numpy as np

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Generate a mesh from a point cloud using Alpha Shapes.")
parser.add_argument('--input', type=str, help='Path to the input PLY file', required=True)
parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter. A smaller value creates a tighter mesh but may have holes. If None, a default is calculated.')
args = parser.parse_args()

# --- Load Point Cloud ---
print(f"Loading point cloud from {args.input}...")
pcd = o3d.io.read_point_cloud(args.input)

if not pcd.has_points():
    print("Error: Point cloud is empty.")
    exit()

# --- Pre-processing: Outlier Removal ---
# This is often a good step to clean up noise before meshing.
print("Removing statistical outliers...")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print("Outliers removed.")

# --- Alpha Shape Reconstruction ---
# Determine the alpha value.
alpha = args.alpha
if alpha is None:
    # If no alpha is provided, calculate a reasonable default.
    # This is a heuristic: a multiple of the average distance between points.
    # You will likely need to tune this multiplier.
    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    alpha = avg_dist * 10
    print(f"Alpha not provided. Using a calculated default: {alpha:.6f}")

print(f"Running Alpha Shape reconstruction with alpha = {alpha}...")
# Note: Alpha Shapes do not require normals.
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

# The resulting mesh is often composed of multiple disconnected components.
# We can get the largest connected component for a cleaner result.
print("Clustering connected components of the mesh...")
triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)

# Get the largest cluster
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh.remove_triangles_by_mask(triangles_to_remove)
print("Kept the largest connected component.")


# --- NEW: Deep Clean and Repair the Mesh ---
# This block fixes subtle issues like degenerate or duplicate triangles
# that can cause the watertight check to fail without obvious errors.
print("\nPerforming deep clean and repair on the mesh...")
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()
print("Mesh repair complete.")


# --- Check for Watertightness and Calculate Volume ---
is_watertight = mesh.is_watertight()

if is_watertight:
    print("\nSuccess: The generated mesh is watertight.")
    volume = mesh.get_volume()
    print(f"The volume of the mesh is: {volume:.6f} cubic units.")
    # Visualize the final watertight mesh
    print("Visualizing the watertight mesh...")
    o3d.visualization.draw_geometries([mesh])
else:
    print("\nWarning: The generated mesh is NOT watertight.")
    print("Volume calculation is not reliable on a non-manifold mesh.")
    
    # --- Visualize all non-manifold issues ---
    geometries_to_draw = [mesh]
    
    # 1. Check for boundary edges (holes) and color them RED
    boundary_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True))
    if len(boundary_edges) > 0:
        print(f"Found {len(boundary_edges)} boundary edges (holes). Highlighting them in RED.")
        lineset_red = o3d.geometry.LineSet(
            points=mesh.vertices,
            lines=o3d.utility.Vector2iVector(boundary_edges)
        )
        lineset_red.paint_uniform_color([1.0, 0.0, 0.0])
        geometries_to_draw.append(lineset_red)
    else:
        print("No boundary edges (holes) found.")

    # 2. Check for other non-manifold edges (>2 faces per edge) and color them GREEN
    non_manifold_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    if len(non_manifold_edges) > 0:
        print(f"Found {len(non_manifold_edges)} non-manifold edges (intersections). Highlighting them in GREEN.")
        lineset_green = o3d.geometry.LineSet(
            points=mesh.vertices,
            lines=o3d.utility.Vector2iVector(non_manifold_edges)
        )
        lineset_green.paint_uniform_color([0.0, 1.0, 0.0])
        geometries_to_draw.append(lineset_green)
    else:
        print("No non-manifold edges found.")

    # 3. Check for non-manifold vertices and color them BLUE
    non_manifold_vertices = np.asarray(mesh.get_non_manifold_vertices())
    if len(non_manifold_vertices) > 0:
        print(f"Found {len(non_manifold_vertices)} non-manifold vertices (pinch points). Highlighting them with BLUE spheres.")
        # Create a small sphere at each non-manifold vertex
        for v_idx in non_manifold_vertices:
            vert = mesh.vertices[v_idx]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=alpha * 0.05) # Sphere size relative to alpha
            sphere.translate(vert)
            sphere.paint_uniform_color([0.0, 0.0, 1.0])
            geometries_to_draw.append(sphere)
    else:
        # This case means the mesh is non-watertight for a very obscure reason,
        # or there's a bug in Open3D's checks.
        print("No non-manifold vertices found. The cause of the non-watertightness is unclear.")


    print("\nVisualizing the mesh with its topological errors highlighted...")
    o3d.visualization.draw_geometries(geometries_to_draw)


# --- Save the Output Mesh ---
output_path = args.input.replace("points.ply", f"alpha_mesh_{alpha:.3f}.ply")
o3d.io.write_triangle_mesh(output_path, mesh)
print(f"Mesh saved to {output_path}")

# --- Visualize ---
# The visualization is now handled inside the if/else block to show the holes.
# print("Visualizing the generated mesh...")
# o3d.visualization.draw_geometries([mesh])