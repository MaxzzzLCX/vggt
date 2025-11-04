eval "$(conda shell.bash hook)"
conda activate vggt

cd "/Users/maxlyu/Documents/vggt"

python parameterization/MeshVolume.py \
    --mesh_path "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_apple/textured.obj" \
    --point_path "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_apple/sparse_sam_scaled_conf0.0/points.ply" \
    --visualize