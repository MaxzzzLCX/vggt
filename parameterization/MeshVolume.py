import open3d as o3d
from pathlib import Path
import numpy as np
import pyransac3d as pyrsc
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Mesh Volume Estimation")
    parser.add_argument("--visualize", action="store_true", help="Visualize the reconstructed mesh")
    return parser.parse_args()


def poisson_reconstruction(point_path, depth=5, visualize=False):
    """
    Perform Poisson surface reconstruction on the given point cloud. 
    Return the volume of the reconstructed mesh if it is watertight.
    """

    pcd = o3d.io.read_point_cloud(point_path)

    # Check if the pcd has normals
    if not pcd.has_normals():
        print("Point cloud has no normals. Estimating normals...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist * 2
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Visualize the mesh
    if visualize:
        o3d.visualization.draw_geometries([mesh])

    volume = None

    # Get the volume
    if mesh.is_watertight():
        volume = mesh.get_volume()
        # print(f"Volume of the reconstructed mesh: {volume} m^3")
    else:
        print("The reconstructed mesh is not watertight.")

        # Check
        non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        non_manifold_vertices = mesh.get_non_manifold_vertices()
        
        # Check for boundary edges (holes)
        boundary_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)

        print(f"Non-manifold edges found: {len(non_manifold_edges)}")
        print(f"Non-manifold vertices found: {len(non_manifold_vertices)}")
        print(f"Boundary edges (holes) found: {len(boundary_edges)}")

        if len(non_manifold_edges) > 0:
            print("Attempting to resolve non-manifold edges by splitting them...")
        
        
            # Remove UVs to allow the function to work ---
            if mesh.has_triangle_uvs():
                print("Temporarily removing texture coordinates (UVs) to fix geometry.")
                mesh.triangle_uvs = o3d.utility.Vector2dVector([])
            
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.remove_degenerate_triangles()

            print(f"After removal, mesh has {len(mesh.get_non_manifold_edges(allow_boundary_edges=False))} non-manifold edges.")
            print(f"After removal, mesh watertight is {mesh.is_watertight()}.")

    return volume if volume else None


def ransac_estimation(point_path, primitive, thresh=None, maxIteration=1000, visualize=False):
    """
    Estimate a mesh from the point cloud using RANSAC.
    - Primitives can be 'sphere', 'cuboid', or 'cylinder'.
    Return the volume of the estimated mesh.
    """

    pcd = o3d.io.read_point_cloud(point_path)
    points = np.asarray(pcd.points) # Convert point cloud to numpy array

    # If not specified, calculate from average distance between points
    if thresh is None:
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        multipler = 2
        thresh = avg_dist * multipler
        print(f"Threshold not provided. Using a calculated default: {thresh:.6f}")


    # Perform Poisson reconstruction as a placeholder for RANSAC-based method
    if primitive == 'sphere':
        sphere = pyrsc.Sphere()
        center, radius, inliers = sphere.fit(points, thresh=thresh, maxIteration=maxIteration)

        # Visualize the point cloud with the fitted sphere
        if visualize:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh.translate(center)
            o3d.visualization.draw_geometries([pcd, mesh], mesh_show_wireframe=True)

        # Calculate Volume
        volume = (4/3) * np.pi * (radius ** 3)

    elif primitive == 'cuboid':
        cuboid = pyrsc.Cuboid()
        # The original `fit` function returns 3 plane equations and inliers
        planes, inliers = cuboid.fit(points, thresh=thresh, maxIteration=maxIteration)

        if len(planes) == 0:
            print("Cuboid RANSAC failed to find a model.")
            return None

        # --- DERIVE BOUNDED CUBOID FROM 3 ORTHOGONAL PLANES ---

        # 1. Extract the three orthogonal normal vectors
        n1 = planes[0][:3]
        n2 = planes[1][:3]
        n3 = planes[2][:3]
        
        # 2. Get the actual 3D points of the inliers
        inlier_points = points[inliers]

        # 3. Project the inlier points onto each normal to find the cuboid's extent
        proj1 = np.dot(inlier_points, n1)
        proj2 = np.dot(inlier_points, n2)
        proj3 = np.dot(inlier_points, n3)

        # 4. Calculate the size (width, height, depth) of the cuboid
        size1 = np.max(proj1) - np.min(proj1)
        size2 = np.max(proj2) - np.min(proj2)
        size3 = np.max(proj3) - np.min(proj3)
        size = np.array([size1, size2, size3])

        # 5. Calculate the center of the cuboid
        center1 = (np.max(proj1) + np.min(proj1)) / 2
        center2 = (np.max(proj2) + np.min(proj2)) / 2
        center3 = (np.max(proj3) + np.min(proj3)) / 2
        center = center1 * n1 + center2 * n2 + center3 * n3

        # 6. Define the rotation matrix from the orthogonal normals
        # The normals form the axes of the cuboid's local coordinate system
        rotation = np.array([n1, n2, n3]).T

        # Visualize the point cloud with the fitted cuboid
        if visualize:
            mesh = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
            # The center of the box created by create_box is at (width/2, height/2, depth/2)
            # We need to translate it to its origin before rotating
            mesh.translate(-size / 2) 
            mesh.rotate(rotation, center=(0,0,0))
            mesh.translate(center)
            o3d.visualization.draw_geometries([pcd, mesh], mesh_show_wireframe=True)
        
        # Calculate Volume
        volume = size[0] * size[1] * size[2]

    elif primitive == 'cylinder':
        raise ValueError("Cylinder RANSAC is not possible due to no height estimation.")

    else:
        raise NotImplementedError(f"Primitive {primitive} not implemented.")


    return volume



def fix_and_get_volume(path):
    """
    Load a mesh .obj file, fix its non-watertight issue, and return its volume.
    """

    mesh = o3d.io.read_triangle_mesh(path)

    # Remove texture coordinates if present
    if mesh.has_triangle_uvs():
            mesh.triangle_uvs = o3d.utility.Vector2dVector([])

    # Fix non-watertight issues
    if not mesh.is_watertight():
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    volume = mesh.get_volume()
    return volume

def single_compare_reconstruction_methods(mesh_path, point_path):

    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-30-meatloaf-338g/textured.obj"
    # point_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-30-meatloaf-338g/point_cloud.ply"
    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-11-red-apple-145g/textured.obj"
    # point_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-11-red-apple-145g/point_cloud.ply"
    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-68-chicken-leg-36g/poly.obj"
    # point_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-68-chicken-leg-36g/point_cloud.ply"
    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-23-captain-crunch-granola-bar-27g/textured.obj"
    # point_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-23-captain-crunch-granola-bar-27g/point_cloud.ply"

    volume = fix_and_get_volume(mesh_path)

    psr_volumes = []
    for depth in [3, 4, 5, 6, 7]:
        psr_volume = poisson_reconstruction(point_path, depth=depth)
        psr_volumes.append(psr_volume)


    print(f"GT Volume of the mesh: {volume} m^3")
    for i, psr_volume in enumerate(psr_volumes):
        if psr_volume:
            print(f"Estimated volume from Poisson reconstruction (depth {i+3}): {psr_volume} m^3; Error {(abs(volume - psr_volume)/volume*100):.2f}%")
        else:
            print(f"Poisson reconstruction (depth {i+3}) NON-WATERTIGHT.")

    # RANSAC estimation    
    for primitive in ['sphere', 'cuboid']:
        ransac_volume = ransac_estimation(point_path, primitive)
        print(f"Estimated volume from RANSAC estimation ({primitive}): {ransac_volume} m^3; Error {(abs(volume - ransac_volume)/volume*100):.2f}%")


def compare_reconstruction_methods(mesh_path, point_path, n_trials=5, visualize=False):
    """
    Compare the different volume estimation methods. 
    For RANSAC, estimate using the mean and var across mutiple trials.
    """

    volume = fix_and_get_volume(mesh_path)

    psr_volumes = []
    for depth in [3, 4, 5, 6, 7]:

        psr_volume = poisson_reconstruction(point_path, depth=depth, visualize=visualize)
        psr_volumes.append(psr_volume)


    print(f"Volume of the mesh: {volume} m^3")
    for i, psr_volume in enumerate(psr_volumes):
        if psr_volume:
            print(f"Estimated volume from Poisson reconstruction (depth {i+3}): {psr_volume} m^3; Error {(abs(volume - psr_volume)/volume*100):.2f}%")
        else:
            print(f"Poisson reconstruction (depth {i+3}) NON-WATERTIGHT.")

    # RANSAC estimation    
    means = []
    vars = []
    for primitive in ['sphere', 'cuboid']:
        mean, var = variance_in_ransac_volume_estimation(mesh_path, point_path, primitive, n_trials=n_trials, visualize=visualize)
        means.append(mean)
        vars.append(var)
        print(f"Estimated volume from RANSAC estimation ({primitive}): {mean} m^3; Error {(abs(volume - mean)/volume*100):.2f}%")
        print(f"Variance in volume estimation: {var}")

    return volume, psr_volumes, means, vars

def variance_in_ransac_volume_estimation(mesh_path, point_path, primitive, n_trials=5, visualize=False):
    """
    Estimate the variance in volume estimation using the specified reconstruction method.
    """
    volumes = []
    for _ in range(n_trials):
        volume = ransac_estimation(point_path, primitive=primitive, visualize=visualize)
        if volume:
            volumes.append(volume)

    if len(volumes) == 0:
        print("All trials resulted in non-watertight meshes.")
        return None, None

    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)

    print(f"Mean estimated volume from RANSAC over {n_trials} trials: {mean_volume} m^3")
    print(f"Standard deviation of estimated volume: {std_volume} m^3")
    return mean_volume, std_volume


def main():
    args = parse_args() # Keep this for when running MeshVolume.py directly
    mesh_file = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-101-steak-piece-28g/poly.obj"
    point_file = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/sparse_sam_conf0.0/points.ply"
    compare_reconstruction_methods(mesh_file, point_file, n_trials=5, visualize=args.visualize)

if __name__ == "__main__":
    main()