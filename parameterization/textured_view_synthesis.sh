eval "$(conda shell.bash hook)"
conda activate vggt

cd /Users/maxlyu/Documents/vggt

# Run the script
python parameterization/view_synthesis_textured.py \
    --mesh "/Users/maxlyu/Documents/nutritionverse-3d-dataset-objectcapture/id-46-costco-salad-sushi-roll-3-29g/poly.obj" \
    --out_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset-objectcapture/id-46-costco-salad-sushi-roll-3-29g/images" \
    --rotation_axis y \
    --radius 5 \
    --num_view 12


