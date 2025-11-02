import numpy as np

def load_depthmap(depth_path):
    """Load a depth map from a .npy file."""
    return np.load(depth_path)

if __name__ == "__main__":
    depth_path = "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/images/view_000_depth.npy"
    
    # Example usage
    depth_map = load_depthmap(depth_path)
    print("Depth map shape:", depth_map.shape)
    print("Depth map data (sample):", depth_map[220, 220])  # Print a small sample of the depth map