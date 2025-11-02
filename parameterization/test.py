import open3d as o3d
from pathlib import Path
import numpy as np

def test_watertight(dataset_folder):
    """
    Test the proportion of watertight meshes in the dataset folder.
    """

    corrupted_files = ["id-66-half-shrimp-salad-roll-69g", "id-90-chicken-wing-33g", "id-63-meatball-37g", "id-96-corn-81g", "id-99-corn-60g"]

    dataset_path = Path(dataset_folder)
    non_watertight_count_pre = 0
    non_watertight_count_post = 0
    non_watertight_files_post = []
    watertight_files = []
    total_count = 0

    for folder in dataset_path.glob("id-*"):
        if folder.is_dir() and folder.name not in corrupted_files:

            object_id = folder.name.split("-")[1]

            # Get the .obj file
            obj_files = list(folder.glob("*.obj"))
            obj_file = obj_files[0]

            mesh = o3d.io.read_triangle_mesh(str(obj_file))
            if not mesh.is_watertight():
                print(f"Mesh {obj_file} is not watertight.")
                non_watertight_count_pre += 1

                # Try basic fixes of removing duplicated vertices
                mesh.remove_duplicated_vertices()
                if not mesh.is_watertight():
                    non_watertight_count_post += 1
                    non_watertight_files_post.append(str(obj_file))
                else:
                    watertight_files.append(str(obj_file))
            
            total_count += 1
        
        else:
            print(f"Error loading mesh in folder {folder.name}.")
            


    ratio_pre = non_watertight_count_pre / total_count
    ratio_post = non_watertight_count_post / total_count
    print(f"Total meshes: {total_count}, Non-watertight meshes (pre-fix): {non_watertight_count_pre}, Ratio: {ratio_pre:.2f}")
    print(f"Non-watertight meshes (post-fix): {non_watertight_count_post}, Ratio: {ratio_post:.2f}")
    print(f"Some watertight files: {(watertight_files[:10])}")
    return ratio_post, total_count-non_watertight_count_post, watertight_files

def get_volume(path):
    """
    Load a mesh .obj file and return its volume.
    """

    mesh = o3d.io.read_triangle_mesh(path)
    volume = mesh.get_volume()
    return volume

def diagnose_non_watertight(mesh_path):
    """
    Loads a mesh and visualizes the specific edges that make it non-watertight.
    """
    print(f"\n--- Diagnosing {mesh_path} ---")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertices():
        print("Error: Mesh is empty.")
        return

    # Check for the most common issue: non-manifold edges
    # An edge is non-manifold if it is shared by more than 2 faces.
    non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    non_manifold_vertices = mesh.get_non_manifold_vertices()
    # non_manifold_triangles = mesh.get_non_manifold_triangles()
    
    # Check for boundary edges (holes)
    boundary_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    
    # The set of boundary edges includes all non-manifold edges, so we can subtract
    # to find edges that are ONLY on the boundary.
    boundary_edges_only = np.setdiff1d(np.asarray(boundary_edges), np.asarray(non_manifold_edges))


    print(f"Found {len(non_manifold_edges)} non-manifold edges (shared by >2 faces).")
    print(f"Found {len(non_manifold_vertices)} non-manifold vertices.")
    # print(f"Found {len(non_manifold_triangles)} non-manifold triangles.")
    print(f"Found {len(boundary_edges_only)} boundary edges (holes, shared by 1 face).")

    if len(non_manifold_edges) == 0 and len(boundary_edges_only) == 0:
        print("No non-manifold or boundary edges found. The issue might be related to duplicate vertices or other topological errors.")
        return

    # Try solving the non-manifold edge
    if len(non_manifold_edges) > 0:
        print("Attempting to resolve non-manifold edges by splitting them...")
        
        # Remove UVs to allow the function to work ---
        if mesh.has_triangle_uvs():
            print("Temporarily removing texture coordinates (UVs) to fix geometry.")
            mesh.triangle_uvs = o3d.utility.Vector2dVector([])
        
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        print(f"After removal, mesh has {len(mesh.get_non_manifold_edges(allow_boundary_edges=False))} non-manifold edges.")
        print(f"After removal, mesh watertight is {mesh.is_watertight()}.")



def main():
    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-1-salad-chicken-strip-7g/textured.obj"
    # volume = get_volume(mesh_path)
    # print(f"Volume of the mesh: {volume}")

    dataset_folder = "/Users/maxlyu/Documents/nutritionverse-3d-dataset"
    ratio_post, watertight_count_post, watertight_files = test_watertight(dataset_folder)
    watertight_files.sort()

    store_file = dataset_folder + "/watertight_files.txt"
    with open(store_file, "w") as f:
        f.write(f"# Total watertight meshes after fix: {watertight_count_post}\n")
        f.write(f"# Ratio of watertight meshes after fix: {ratio_post:.2f}\n")
        for item in watertight_files:
            f.write("%s\n" % item)
    
    # mesh_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/id-14-salad-beef-strip-7g/textured.obj"
    # diagnose_non_watertight(mesh_path)

if __name__ == "__main__":
    main()
