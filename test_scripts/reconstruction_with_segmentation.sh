
cd "/Users/maxlyu/Documents/vggt"

eval "$(conda shell.bash hook)"
conda activate vggt

python test_scripts/VGGT_COLMAP.py \
    --scene_dir "/Users/maxlyu/Documents/scenes/apple_kitchen" \
    --mask \
    --mask_dir "/Users/maxlyu/Documents/scenes/apple_kitchen/masks" \
    --conf_thres_value 0 \