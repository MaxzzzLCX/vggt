source /scratch/cl927/miniconda3/etc/profile.d/conda.sh
conda activate vggt

cd /scratch/cl927/vggt
export PYTHONPATH="/scratch/cl927/vggt:${PYTHONPATH:-}"

python test_scripts/VGGT_COLMAP.py \
    --scene_dir "/scratch/cl927/nutritionverse-3d-new/_test_original_vggt_id-11-red-apple-145g" \
    --mask \
    --mask_dir "/scratch/cl927/nutritionverse-3d-new/_test_original_vggt_id-11-red-apple-145g" \
    --conf_thres_value 0 