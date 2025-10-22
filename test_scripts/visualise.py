import open3d as o3d
import argparse

# Input file path
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Path to the input PLY file', required=True)
args = parser.parse_args()

# Load and visualize the point cloud
pcd = o3d.io.read_point_cloud(args.input)  # auto-load colors/normals if present
print(pcd)  # size & fields
o3d.visualization.draw_geometries([pcd])
