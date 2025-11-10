eval "$(conda shell.bash hook)"
conda activate vggt

cd /Users/maxlyu/Documents/vggt

# Run the script
python parameterization/DatasetViewSynthesis.py \
    --dataset_folder "/Users/maxlyu/Documents/nutritionverse-3d-dataset-estimation" \
    --width 518 \
    --height 518 \
    --radius 5 \
    --elevation 0 \
    --rotation_axis y \
    --yfov_deg 50 \
    --start_index 69 \
    --end_index 105
    


