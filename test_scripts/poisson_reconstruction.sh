
cd "/Users/maxlyu/Documents/vggt"

eval "$(conda shell.bash hook)"
conda activate vggt

python3 test_scripts/poisson_mesh_generation_new.py \
    --input "/Users/maxlyu/Documents/scenes/charger/sparse_sam_conf0.0/points.ply" \
    --estimate_base \
    --reconstruction_method "poisson" \
    --depth 4 \
    --real_height 4.6
