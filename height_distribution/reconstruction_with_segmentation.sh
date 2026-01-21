
cd "/Users/maxlyu/Documents/vggt"

eval "$(conda shell.bash hook)"
conda activate vggt

python test_scripts/VGGT_COLMAP.py \
    --scene_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset-vggt_height/test_vggt_steak_101_eight_images" \
    --mask \
    --mask_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset-vggt_height/test_vggt_steak_101_eight_images/masks" \
    --conf_thres_value 0 \
    --scale_pointcloud