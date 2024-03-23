# this process may take much time, but this is necessary if you try the training process
# python3 scripts/create_dataset.py \
# --audio_dir DCM_data/amc_data_split_by_categories/Train/Audio \
# --camera_c_dir DCM_data/amc_data_split_by_categories/Train/CameraCentric \
# --motion_gt_dir DCM_data/amc_data_split_by_categories/Train/Simplified_MotionGlobalTransform \
# --data_dir DCM_data/amc_data_split_by_categories/Train \
# --extract_jukebox

# python3 scripts/create_dataset.py \
# --audio_dir DCM_data/amc_data_split_by_categories/Validation/Audio \
# --camera_c_dir DCM_data/amc_data_split_by_categories/Validation/CameraCentric \
# --motion_gt_dir DCM_data/amc_data_split_by_categories/Validation/Simplified_MotionGlobalTransform \
# --data_dir DCM_data/amc_data_split_by_categories/Validation \
# --extract_jukebox

# run this command if you just want to test the model
python3 scripts/create_dataset.py \
--audio_dir DCM_data/amc_data_split_by_categories/Test/Audio \
--camera_c_dir DCM_data/amc_data_split_by_categories/Test/CameraCentric \
--motion_gt_dir DCM_data/amc_data_split_by_categories/Test/Simplified_MotionGlobalTransform \
--data_dir DCM_data/amc_data_split_by_categories/Test \
--extract_jukebox