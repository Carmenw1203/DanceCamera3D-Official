# align the music audio, motion and camera raw data
python3 scripts/amc_align.py
# interpolate the camera keyframe data
python3 scripts/camera_preprocess.py
# split raw data to sub-sequences
python3 scripts/split_long_data.py --split_from_file True
# split sub-sequences into train, test validation sets according to categories
python3 scripts/split_by_categories.py --split_from_file True