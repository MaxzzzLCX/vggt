eval "$(conda shell.bash hook)"
conda activate vggt

cd "/Users/maxlyu/Documents/vggt"

python viewpoints/HeightEstimationExtrinsic.py \
    --depth /Users/maxlyu/Documents/nutritionverse-3d-dataset/test_perspectives_orthogonal/images \
    --mask /Users/maxlyu/Documents/nutritionverse-3d-dataset/test_perspectives_orthogonal/masks \
    --K_json /Users/maxlyu/Documents/nutritionverse-3d-dataset/test_perspectives_orthogonal/images/cameras.json \

