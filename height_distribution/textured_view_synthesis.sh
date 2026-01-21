eval "$(conda shell.bash hook)"
conda activate vggt

cd /Users/maxlyu/Documents/vggt

# Run the script
python parameterization/view_synthesis_textured.py \
    --mesh "/Users/maxlyu/Documents/nutritionverse-3d-dataset-vggt_height/test_vggt_steak_101_eight_images/poly.obj" \
    --out_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset-vggt_height/test_vggt_steak_101_eight_images/images" \
    --rotation_axis y \
    --radius 5 \
    --num_view 8


