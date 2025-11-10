eval "$(conda shell.bash hook)"
conda activate vggt

cd /Users/maxlyu/Documents/vggt

# Run the script
python parameterization/orthogonal_view_synthesis.py \
    --mesh "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_chicken_leg_27/textured.obj" \
    --out_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_chicken_leg_27/images" \
    --rotation_axis y \
    --radius 5 \
    --elevation 0


