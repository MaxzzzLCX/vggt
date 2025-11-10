eval "$(conda shell.bash hook)"
conda activate vggt

cd "/Users/maxlyu/Documents/vggt"

object_folder="test_chicken_leg_27"

python viewpoints/HeightEstimationIntrinsic.py \
    --depth /Users/maxlyu/Documents/nutritionverse-3d-dataset/${object_folder}/images \
    --mask /Users/maxlyu/Documents/nutritionverse-3d-dataset/${object_folder}/masks \
    --K_json /Users/maxlyu/Documents/nutritionverse-3d-dataset/${object_folder}/images/cameras.json \
    --view_index 0
