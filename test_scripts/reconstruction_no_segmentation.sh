
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

eval "$(conda shell.bash hook)"
conda activate vggt
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python test_scripts/VGGT_COLMAP.py \
    --scene_dir "/Users/maxlyu/Documents/nutritionverse-3d-dataset/test_syn_2" \
    --conf_thres_value 0