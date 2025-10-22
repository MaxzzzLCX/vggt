import open3d as o3d
import argparse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

# Seed all randomness
SEED = 42
np.random.seed(SEED)
o3d.utility.random.seed(SEED)


# Input and output files
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to the input PLY file', required=True)
parser.add_argument('--depth', type=int, default=8, help='Depth parameter for Poisson reconstruction (higher = more detail, but slower)')
parser.add_argument('--density_threshold', type=float, default=0, help='Density threshold for removing low-density vertices (between 0 and 1)')
parser.add_argument('--estimate_base', action='store_true', help='Whether to estimate and align to base plane automatically')
parser.add_argument('--reconstruction_method', type=str, default='poisson', help='Reconstruction method to use')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.input)

# --- STEP 1: VOXEL DOWN-SAMPLING ---
# This cleans up the "silhouette" effect by merging dense points.
# The voxel_size is in the same units as your point cloud. You may need to tune it.
# A larger voxel_size means more aggressive downsampling.
voxel_size = 0.01 # Assuming your object is a few units in size. Adjust as needed.
print(f"Original point count: {len(pcd.points)}")
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"Down-sampled to: {len(pcd.points)} points")

# Save the down-sampled point cloud for inspection
downsampled_output_path = args.input.replace("points.ply", "points_downsampled.ply")
o3d.io.write_point_cloud(downsampled_output_path, pcd)


# --- STEP 2: OUTLIER REMOVAL ---
# This is still useful after downsampling to remove stray points.
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
removed_output_path = args.input.replace("points.ply", "points_removed_outliers.ply")
o3d.io.write_point_cloud(removed_output_path, pcd)
print(f"Points remaining after outlier removal: {len(pcd.points)}")


# --- STEP 3: NORMAL ESTIMATION ---
# Estimate normals on the cleaned, down-sampled point cloud.
print("Estimating normals...")
radius = np.mean(pcd.compute_nearest_neighbor_distance()) * 2 # Use a smaller radius multiplier now
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
# Orient normals based on local neighborhood
pcd.orient_normals_consistent_tangent_plane(k=100) # k is the number of neighbors to check

# pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., -1., 0.]))

pcd_combined = pcd

# # --- DIAGNOSTIC STEP: VISUALIZE THE FINAL POINT CLOUD ---
# print("Visualizing the point cloud after attaching the base...")
# o3d.visualization.draw_geometries([pcd_combined], point_show_normal=True)

if args.estimate_base:

    # --- NEW STEP: AUTOMATICALLY FIND THE BASE PLANE USING RANSAC ---
    print("Automatically finding the base plane using RANSAC...")
    # RANSAC finds the largest plane in the point cloud.
    # distance_threshold: how close a point must be to the plane to be considered an inlier.
    # Tune this based on the noise level of your point cloud near the base.

    distance_threshold = voxel_size * 1.5

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    base_normal = np.array([a, b, c])

    # Ensure the normal points "up" relative to the point cloud's center
    pcd_center = pcd.get_center()
    if np.dot(base_normal, pcd_center - (base_normal * -d)) > 0:
        base_normal = -base_normal
    print(f"Detected base plane normal: {base_normal}")


    # --- NEW STEP 3: ALIGN POINT CLOUD TO THIS BASE NORMAL ---
    print("Aligning point cloud to the detected base...")
    # Target normal is the Z-axis [0, 0, 1]
    target_normal = np.array([0, 0, 1.0])
    # Get the rotation matrix required to align the base_normal with the target_normal
    R = Rotation.align_vectors([target_normal], [base_normal])[0].as_matrix()
    pcd.rotate(R, center=(0,0,0))


    # --- STEP 5: GENERATE A FLAT BASE (Your logic, now works correctly) ---
    print("Generating a flat base to close the mesh...")
    lowest_z = pcd.get_min_bound()[2]
    points_xy = np.asarray(pcd.points)[:, [0, 1]]
    # ... (The rest of your base generation logic is now correct and can be used as-is)
    min_bound_xy = np.min(points_xy, axis=0)
    max_bound_xy = np.max(points_xy, axis=0)
    grid_res = 100
    x_range = np.linspace(min_bound_xy[0], max_bound_xy[0], grid_res)
    y_range = np.linspace(min_bound_xy[1], max_bound_xy[1], grid_res)
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    base_points_xy = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    from matplotlib.path import Path
    hull = ConvexHull(points_xy)
    path = Path(points_xy[hull.vertices])
    inside_mask = path.contains_points(base_points_xy)
    base_points_xy_filtered = base_points_xy[inside_mask]
    base_points_3d = np.hstack([base_points_xy_filtered, np.full((len(base_points_xy_filtered), 1), lowest_z)])
    base_normals = np.tile([0, 0, -1], (len(base_points_3d), 1)) # Normals point "out" in -Z direction
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(base_points_3d)
    pcd_base.normals = o3d.utility.Vector3dVector(base_normals)


    print(f"Generated {len(pcd_base.points)} points for the base.")
    pcd_combined = pcd + pcd_base
    print(f"Total points after adding base: {len(pcd_combined.points)}")



# --- DIAGNOSTIC STEP: VISUALIZE THE FINAL POINT CLOUD ---
print("Visualizing the final point cloud before meshing...")
o3d.visualization.draw_geometries([pcd_combined], point_show_normal=True)


if args.reconstruction_method == 'poisson':
    # --- STEP 5: POISSON SURFACE RECONSTRUCTION ---
    # Now, run Poisson on the much cleaner, pre-filtered point cloud.
    print("Running Poisson surface reconstruction...")
    # --- FIX: Use the pcd_combined which includes the base ---
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_combined, depth=args.depth)

elif args.reconstruction_method == 'alpha':
    # --- ALPHA SHAPE RECONSTRUCTION AS AN ALTERNATIVE ---
    alpha = 0.1  # You may need to tune this parameter
    print(f"Running Alpha Shape reconstruction with alpha = {alpha}...")
    # Note: Alpha Shapes do not require normals.
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_combined, alpha)
else:
    raise NotImplementedError(f"Reconstruction method '{args.reconstruction_method}' is not implemented.")

print("Cleaning up the mesh...")
bbox = pcd_combined.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# Correct the point clouds
mesh = mesh.merge_close_vertices(1e-2)
mesh.orient_triangles()

# This is the most important step for ensuring watertightness
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()
print("Mesh repaired.")


mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
mesh = mesh.fill_holes(100000)
mesh = mesh.to_legacy()
# mesh.remove_degenerate_triangles()
# mesh.remove_duplicated_vertices()
# mesh.remove_duplicated_triangles()
# mesh.remove_unreferenced_vertices()


# DEBUG: Find boundary vertices




# Remove vertices of low density (have low support)
if args.density_threshold > 0:
    vertices_to_remove = densities < np.quantile(densities, args.density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

output_path = args.input.replace("points.ply", f"poisson_mesh_depth{args.depth}.ply")
o3d.io.write_triangle_mesh(output_path, mesh)
print(f"Mesh saved to {output_path}")

# Check if mesh is watertight
if mesh.is_watertight():
    print("The generated mesh is watertight.")
    print(f"Volume: {mesh.get_volume()}")
else:
    print("The generated mesh is NOT watertight.")

# Visualize
print("Visualizing the generated mesh...")
o3d.visualization.draw_geometries([mesh])

# # Fix non-watertight meshes
# hole_size = 1000000
# mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
# mesh_t = mesh_t.fill_holes(hole_size)

# print("Visualizing the mesh after hole filling...")
# o3d.visualization.draw_geometries([mesh_t.to_legacy()])
# output_path_filled = args.input.replace("points.ply", f"poisson_mesh_depth{args.depth}_filled.ply")
# o3d.io.write_triangle_mesh(output_path_filled, mesh_t.to_legacy())

# if mesh_t.to_legacy().is_watertight():
#     print("The filled mesh is watertight.")
# else:
#     print("The filled mesh is NOT watertight.")