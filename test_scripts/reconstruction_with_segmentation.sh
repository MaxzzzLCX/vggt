
cd "/Users/maxlyu/Documents/vggt"

eval "$(conda shell.bash hook)"
conda activate vggt

python test_scripts/VGGT_COLMAP.py \
    --scene_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak" \
    --mask \
    --mask_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/masks" \
    --conf_thres_value 0 \