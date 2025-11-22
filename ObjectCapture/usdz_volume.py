import trimesh
import subprocess
from pathlib import Path
import numpy as np

def diagnose_watertight(mesh: trimesh.Trimesh):
    """Diagnose and print watertight issues of a mesh."""
    print(f"Is watertight: {mesh.is_watertight}")
    if not mesh.is_watertight:
        print("\n========== Watertightness Diagnostics ==========")
        print(f"Is watertight: {mesh.is_watertight}")
        print(f"Is volume mode: {mesh.is_volume}")
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        
        # Euler characteristic (should be 2 for closed surface)
        euler = mesh.euler_number
        print(f"Euler number: {euler} (should be 2 for closed surface)")
        
        # Check face winding consistency
        try:
            mesh.fix_normals()
            print(f"Face winding: consistent after fix_normals()")
        except Exception as e:
            print(f"Face winding: inconsistent ({e})")
        
        # Find boundary edges (edges belonging to only 1 face)
        edges = mesh.edges_unique
        edge_face_count = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        boundary_edges = edges[edge_face_count == 1]
        print(f"Boundary edges (holes): {len(boundary_edges)} (should be 0)")
        
        # Non-manifold edges (shared by >2 faces)
        nonmanifold_edges = edges[edge_face_count > 2]
        print(f"Non-manifold edges: {len(nonmanifold_edges)} (should be 0)")
        
        # Degenerate faces (zero area)
        areas = mesh.area_faces
        degenerate = (areas < 1e-10).sum()
        print(f"Degenerate faces: {degenerate} (should be 0)")
        
        # Duplicate faces
        unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
        duplicates = len(mesh.faces) - len(unique_faces)
        print(f"Duplicate faces: {duplicates} (should be 0)")
        
    

def read_volume_usdz(usdz_path: str, fix: str):
    """Convert USDZ to OBJ via Blender, then compute volume with Trimesh."""
    usdz_path = Path(usdz_path)
    obj_path = usdz_path.with_suffix('.obj')
    
    # Find Blender executable (common macOS paths)
    blender_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/local/bin/blender",
        "blender"  # fallback if in PATH
    ]
    blender_exe = None
    for p in blender_paths:
        if Path(p).exists() or p == "blender":
            blender_exe = p
            break
    
    if not blender_exe:
        raise FileNotFoundError("Blender not found. Install: brew install --cask blender")
    
    # Convert USDZ → OBJ using Blender (clear default scene first)
    script = f'''
import bpy
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
bpy.ops.wm.usd_import(filepath="{usdz_path}")
bpy.ops.wm.obj_export(filepath="{obj_path}", export_triangulated_mesh=True)
'''
    result = subprocess.run([blender_exe, "--background", "--python-expr", script], 
                          check=True, capture_output=True, text=True)
    
    if not obj_path.exists():
        print("Blender stderr:", result.stderr)
        raise RuntimeError(f"Blender failed to create {obj_path}")
    
    # Load OBJ and compute volume
    mesh = trimesh.load(str(obj_path), force='mesh')
    if mesh.is_watertight:
        print(f"Mesh is watertight")
        return mesh.volume
    else:
        print(f"Mesh is not watertight, diagnosing...")
        diagnose_watertight(mesh)

        # Try to fix
        mesh.fill_holes()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        mesh.fill_holes()
        print(f"After repair, is watertight: {mesh.is_watertight}")
        if mesh.is_watertight:
            return mesh.volume
        else:

            convex_hull = mesh.convex_hull
            print(f"Convex hull volume: {convex_hull.volume} m^3")

            pitch = 0.001 # 1mm voxels
            voxelized = mesh.voxelized(pitch)
            voxel_vol = voxelized.volume 
            print(f"Voxelized volume (pitch={pitch}): {voxel_vol} m^3")

            
            if fix == 'convex_hull':
                mesh.show()  # Uncomment to visualize
                return convex_hull.volume
            elif fix == 'voxel':
                voxelized.show()
                return voxel_vol

def main():
    volume = read_volume_usdz("/Users/maxlyu/Desktop/ObjectCapture_iOS_App/PeanutButter_19.usdz", fix='convex_hull')
    print(f"Volume: {volume} m^3")

if __name__ == "__main__":
    main()