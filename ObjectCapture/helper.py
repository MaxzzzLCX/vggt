import trimesh
from pathlib import Path

def usdz_to_obj(usdz_path: str, obj_path: str):
    """Convert USDZ to OBJ using USD Python API."""
    try:
        from pxr import Usd, UsdGeom
        import tempfile
        import shutil
        
        # Extract USDZ (it's a zip) to temp folder
        temp_dir = tempfile.mkdtemp()
        shutil.unpack_archive(usdz_path, temp_dir, 'zip')
        
        # Find .usdc or .usd file inside
        usd_files = list(Path(temp_dir).glob('*.usd*'))
        if not usd_files:
            raise FileNotFoundError(f"No USD file found in {usdz_path}")
        
        stage = Usd.Stage.Open(str(usd_files[0]))
        # Export to OBJ (requires usdcat or custom traversal; simple approach: load with trimesh via intermediate format)
        # Better: use external tool or just load directly if trimesh supports USD
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"[WARN] USD→OBJ conversion requires additional tooling. Use usdcat or Blender.")
        return None
    except ImportError:
        print("[ERROR] pxr (USD) not installed. Install: pip install usd-core")
        return None

def read_volume(mesh_path: str):
    """
    Reads a mesh from OBJ/USDZ and computes volume.
    For USDZ: converts to OBJ first (requires external tool).
    """
    p = Path(mesh_path)
    
    # If USDZ, convert first (placeholder; use external tool)
    if p.suffix.lower() == '.usdz':
        print(f"[INFO] USDZ detected. Convert to OBJ first using:")
        print(f"  usdcat -o {p.stem}.obj {mesh_path}")
        print(f"  OR use Blender: File > Import > USD > Export > Wavefront (.obj)")
        raise NotImplementedError("Trimesh cannot load USDZ directly. Convert to OBJ first.")
    
    mesh = trimesh.load(mesh_path, force='mesh')

    print(f"Watertight before repair: {mesh.is_watertight}")
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.merge_vertices()
    mesh.fix_normals()
    try:
        mesh.fill_holes()
    except Exception:
        pass
    raw_volume = mesh.volume
    print(f"Watertight after repair: {mesh.is_watertight}")
    print(f"Raw volume is {raw_volume} cubic units")
    watertight = mesh.is_watertight
    
    mesh = mesh.convex_hull
    convex_volume = mesh.volume
    print(f"Watertight after convex hull: {mesh.is_watertight}")
    print(f"Convex hull volume is {convex_volume} cubic units")
    
    return raw_volume, convex_volume, watertight

def main():

    

    mesh_path = "/Users/maxlyu/Desktop/iOS_ObjectCapture.usdz"  # or .obj
    
    # Convert to obj
    usdz_to_obj(mesh_path, "/Users/maxlyu/Desktop/iOS_ObjectCapture.obj")

    raw_volume, convex_volume, watertight = read_volume(mesh_path)
    print(f"Volume of the mesh: {raw_volume} cubic units")
    print(f"Volume of the convex hull: {convex_volume} cubic units")

if __name__ == "__main__":
    main()

