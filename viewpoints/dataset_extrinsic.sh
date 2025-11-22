eval "$(conda shell.bash hook)"
conda activate vggt

cd "/Users/maxlyu/Documents/vggt"

python viewpoints/TestExtrinsicAll.py \
    --dataset_folder /Users/maxlyu/Documents/nutritionverse-3d-dataset-manual \
    --start_index 0 \
    --end_index 1 \
    --save_csv \
    --save_npy \
    --height_method "nearest"
