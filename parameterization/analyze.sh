cd "/Users/maxlyu/Documents/vggt"

eval "$(conda shell.bash hook)"
conda activate vggt

python parameterization/dataset_statistics.py \
   --store_results \
   --results_folder "./results" \
   --start_index 65 \
   --end_index 76