eval "$(conda shell.bash hook)"
conda activate sam2

cd /Users/maxlyu/Documents/sam2

python3 test_scripts/image_segment_interactive.py \
    --image_folder "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_steak/images" \
    --default_keypoints \
    --save_mask \
    --mask_num 1

