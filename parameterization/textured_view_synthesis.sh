eval "$(conda shell.bash hook)"
conda activate vggt

cd /Users/maxlyu/Documents/vggt

# Run the script
python parameterization/view_synthesis_textured.py \
    --mesh "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/poly.obj" \
    --out_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/images" \
    --rotation_axis y


