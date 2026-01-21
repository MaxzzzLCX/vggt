cd "/Users/maxlyu/Documents/sam2"

eval "$(conda shell.bash hook)"
conda activate sam2

python3 test_scripts/image_segment_interactive.py \
    --image_folder "/Users/maxlyu/Documents/nutritionverse-3d-dataset-vggt_height/test_vggt_steak_101_eight_images/images" \
    --interactive \
    --save_mask \
    --mask_num 1