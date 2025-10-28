import open3d as o3d
import argparse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def scale_analysis(pcd):

    # --- NEW: SCALE ANALYSIS ---
    print("\n" + "="*60)
    print("POINT CLOUD SCALE ANALYSIS")
    print("="*60)

    # Get the bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    # Calculate spans in each direction
    span_x = max_bound[0] - min_bound[0]
    span_y = max_bound[1] - min_bound[1]
    span_z = max_bound[2] - min_bound[2]

    print(f"Bounding box minimum: [{min_bound[0]:.4f}, {min_bound[1]:.4f}, {min_bound[2]:.4f}]")
    print(f"Bounding box maximum: [{max_bound[0]:.4f}, {max_bound[1]:.4f}, {max_bound[2]:.4f}]")
    print(f"\nSpan in X direction: {span_x:.4f} units")
    print(f"Span in Y direction: {span_y:.4f} units")
    print(f"Span in Z direction: {span_z:.4f} units")
    print(f"Average span: {(span_x + span_y + span_z)/3:.4f} units")

    # Calculate the diagonal (overall size)
    diagonal = np.linalg.norm(max_bound - min_bound)
    print(f"Diagonal (corner-to-corner): {diagonal:.4f} units")

    # Calculate average point-to-point distance (sampling for efficiency)
    points = np.asarray(pcd.points)
    if len(points) > 10000:
        # Sample 10000 points for efficiency
        sample_indices = np.random.choice(len(points), 10000, replace=False)
        sample_points = points[sample_indices]
    else:
        sample_points = points

    # Compute nearest neighbor distances
    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sample_points)
    distances = pcd_sample.compute_nearest_neighbor_distance()
    avg_nn_distance = np.mean(distances)
    median_nn_distance = np.median(distances)

    print(f"\nAverage nearest-neighbor distance: {avg_nn_distance:.6f} units")
    print(f"Median nearest-neighbor distance: {median_nn_distance:.6f} units")

    # Suggest voxel size
    suggested_voxel_size = avg_nn_distance * 2
    print(f"\nSuggested voxel_size (2x avg NN distance): {suggested_voxel_size:.6f} units")
    print(f"This would reduce ~{len(pcd.points)} points to approximately {int(len(pcd.points) * (avg_nn_distance / suggested_voxel_size)**3)} points")

    return suggested_voxel_size


def conversion_factor_analysis(pcd, real_height):
    """
    The point cloud is oriented (rotated) such that the base is flat on the XY plane, during base estimation.
    Therefore, we can use the Z span to estimate the height of the object in point cloud units.
    """

    # --- NEW: SCALE ANALYSIS ---
    print("\n" + "="*60)
    print("POINT CLOUD CONVERSION FACTOR ANALYSIS")
    print("="*60)

    # Get the 1 percentile and 99 percentile Z values to avoid outliers
    points = np.asarray(pcd.points)
    print("Total number of points for conversion factor analysis:", len(points))
    z_coords = points[:, 2]
    z_min = np.percentile(z_coords, 0.1)
    z_max = np.percentile(z_coords, 99.9)
    span_z = z_max - z_min
    print(f"1st percentile Z: {z_min:.4f}")
    print(f"99th percentile Z: {z_max:.4f}")
    print(f"Span in Z direction (1st to 99th percentile): {span_z:.4f} units")

    # Calculate conversion factor
    conversion_factor = real_height / span_z
    print(f"Conversion factor: {conversion_factor:.4f} cm/unit")

    return conversion_factor


# Seed all randomness
SEED = 42
np.random.seed(SEED)
o3d.utility.random.seed(SEED)


# Input and output files
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to the input PLY file', required=True)
parser.add_argument('--depth', type=int, default=4, help='Depth parameter for Poisson reconstruction (higher = more detail, but slower)')
parser.add_argument('--density_threshold', type=float, default=0, help='Density threshold for removing low-density vertices (between 0 and 1)')
parser.add_argument('--estimate_base', action='store_true', help='Whether to estimate and align to base plane automatically')
parser.add_argument('--reconstruction_method', type=str, default='poisson', help='Reconstruction method to use')
parser.add_argument('--real_height', type=float, required=True, help='Real height of the object in cm')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.input)

# # Optional scale analysis
# voxel_size = scale_analysis(pcd) # Uses double of average nearest-neighbor distance

# --- STEP 1: VOXEL DOWN-SAMPLING ---
# This cleans up the "silhouette" effect by merging dense points.
# The voxel_size is in the same units as your point cloud. You may need to tune it.
# A larger voxel_size means more aggressive downsampling.
# voxel_size = 0.01 # Assuming your object is a few units in size. Adjust as needed.
# print(f"Original point count: {len(pcd.points)}")
# pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
# print(f"Down-sampled to: {len(pcd.points)} points")

# # Save the down-sampled point cloud for inspection
# downsampled_output_path = args.input.replace("points.ply", "points_downsampled.ply")
# o3d.io.write_point_cloud(downsampled_output_path, pcd)


# --- STEP 2: OUTLIER REMOVAL ---
# This is still useful after downsampling to remove stray points.
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
removed_output_path = args.input.replace("points.ply", "points_removed_outliers.ply")
o3d.io.write_point_cloud(removed_output_path, pcd)
print(f"Points remaining after outlier removal: {len(pcd.points)}")

# TEST
print(f"AFTER OUTLIER REMOVAL:")
# real_object_length = 7 # in cm (roughly diameter of the apple)
voxel_size = scale_analysis(pcd) # Uses double of average nearest-neighbor distance




# --- STEP 3: NORMAL ESTIMATION ---
# Estimate normals on the cleaned, down-sampled point cloud.
print("Estimating normals...")
radius = np.mean(pcd.compute_nearest_neighbor_distance()) * 2 # Use a smaller radius multiplier now
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
# Orient normals based on local neighborhood
pcd.orient_normals_consistent_tangent_plane(k=200) # k is the number of neighbors to check

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

    voxel_size = 0.01 if not voxel_size else voxel_size
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

    # Here, we have rotated the mesh to align the base with xy-plane. 
    # Time to analyze the conversion factor
    conversion_factor = conversion_factor_analysis(pcd, args.real_height)
    print(f"Conversion factor: {conversion_factor:.4f} cm/unit")

    # --- STEP 5: GENERATE A FLAT BASE (Modified to use only bottom points) ---
    print("Generating a flat base to close the mesh...")
    points = np.asarray(pcd.points)
    z_coords = points[:, 2]

    # --- NEW: Only use the bottom 3% of points for the base hull ---
    z_percentile = 3  # Use bottom 3% of points
    z_threshold = np.percentile(z_coords, z_percentile)
    bottom_mask = z_coords <= z_threshold
    bottom_points = points[bottom_mask]

    print(f"Using {len(bottom_points)} points (bottom {z_percentile}%) to define the base footprint.")

    # Now use only the XY coordinates of these bottom points
    points_xy = bottom_points[:, [0, 1]]

    # The lowest Z is still the global minimum
    # lowest_z = np.min(z_coords)

    # Use the 10 percentile of bottom 10% of points as the base Z to avoid outliers
    base_z = np.percentile(z_coords[bottom_mask], 10)

    # --- Rest of the code remains the same, but operates on points_xy from bottom points ---
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

    base_points_3d = np.hstack([base_points_xy_filtered, np.full((len(base_points_xy_filtered), 1), base_z)])
    base_normals = np.tile([0, 0, -1], (len(base_points_3d), 1))

    # --- FIX: Assign a color to the base points ---
    # Let's choose a medium gray color. RGB values are between 0 and 1.
    base_color = [0.5, 0.5, 0.5] 
    base_colors = np.tile(base_color, (len(base_points_3d), 1))

    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(base_points_3d)
    pcd_base.normals = o3d.utility.Vector3dVector(base_normals)
    # --- FIX: Add the color attribute to the base point cloud ---
    pcd_base.colors = o3d.utility.Vector3dVector(base_colors)

    print(f"Generated {len(pcd_base.points)} points for the base.")
    pcd_combined = pcd + pcd_base
    print(f"Total points after adding base: {len(pcd_combined.points)}")



    # --- DIAGNOSTIC STEP: VISUALIZE THE FINAL POINT CLOUD ---
    print("Visualizing the final point cloud before meshing...")
    o3d.visualization.draw_geometries([pcd_combined], point_show_normal=True)

    # Save the combined point cloud
    combined_output_path = args.input.replace("points.ply", "points_with_base.ply")
    o3d.io.write_point_cloud(combined_output_path, pcd_combined)
    print(f"Combined point cloud with base saved to {combined_output_path}")




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

# print("Cleaning up the mesh...")
# bbox = pcd_combined.get_axis_aligned_bounding_box()
# mesh = mesh.crop(bbox)

# Correct the point clouds
mesh = mesh.merge_close_vertices(1e-2)
mesh.orient_triangles()

# This is the most important step for ensuring watertightness
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()
print("Mesh repaired.")

print("Visualizing the mesh before fill_holes...")
o3d.visualization.draw_geometries([mesh])
output_path = args.input.replace("points.ply", f"poisson_mesh_nofill_depth{args.depth}.ply")
o3d.io.write_triangle_mesh(output_path, mesh)
print(f"Mesh saved to {output_path}")

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

output_path = args.input.replace("points.ply", f"poisson_mesh_fillhole_depth{args.depth}.ply")
o3d.io.write_triangle_mesh(output_path, mesh)
print(f"Mesh saved to {output_path}")

# Check if mesh is watertight
if mesh.is_watertight():
    print("The generated mesh is watertight.")
    print(f"Volume: {mesh.get_volume()} units")
    print(f"Conversion factor: {conversion_factor} cm/unit")
    volume_cm3 = mesh.get_volume() * (conversion_factor ** 3)
    print(f"Volume in cubic cm: {volume_cm3} cm³ (or mL)")
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