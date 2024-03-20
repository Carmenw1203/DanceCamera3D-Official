import os
import json
import glob
import librosa as lr
import numpy as np
import soundfile as sf
import tqdm as tqdm

fps = 30

def FrameTimeFunc(e):
    return e["FrameTime"]

def GlobalTransform2Keypoints(gt):
    num_keypoints = int(len(gt['Transform'])/16)
    kps = []
    for kp_i in range(num_keypoints):
        kps += gt['Transform'][kp_i*16+12:kp_i*16+15]
    return kps

def SliceAudio(audio_path, stride, length, max_num_slices, audio_out):
    audio, sr = lr.load(audio_path, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_path))[0]

    start_idx = 0
    window = int(length*sr)
    stride_step = int(stride*sr)
    slice_count = 0

    while start_idx <= len(audio) - window and slice_count < max_num_slices:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{audio_out}/{file_name}_slice{slice_count}.wav", audio_slice, sr)
        start_idx += stride_step
        slice_count += 1
    return slice_count

def SliceCameraCentric(camera_c_path, stride, length, camera_c_out):
    with open(camera_c_path, 'r') as cf:
        camera_data = json.load(cf)
    total_frame = len(camera_data["camera_eye"])
    camera_eye = camera_data["camera_eye"]
    camera_z = camera_data["camera_z"]
    camera_y = camera_data["camera_y"]
    camera_x = camera_data["camera_x"]
    camera_dis = camera_data["Distance"]
    camera_pos = camera_data["Position"]
    camera_rot = camera_data["Rotation"]
    camera_fov = camera_data["Fov"]
    file_name = os.path.splitext(os.path.basename(camera_c_path))[0]

    start_idx = 0
    window = int(length*fps)
    stride_step = int(stride*fps)
    slice_count = 0

    while start_idx <= total_frame - window:
        camera_eye_slices, camera_z_slices, camera_y_slices, camera_x_slices, camera_dis_slices, camera_pos_slices, camera_rot_slices, camera_fov_slices = (
            camera_eye[start_idx : start_idx + window],
            camera_z[start_idx : start_idx + window],
            camera_y[start_idx : start_idx + window],
            camera_x[start_idx : start_idx + window],
            camera_dis[start_idx : start_idx + window],
            camera_pos[start_idx : start_idx + window],
            camera_rot[start_idx : start_idx + window],
            camera_fov[start_idx : start_idx + window]
        )
        assert len(camera_eye_slices) == len(camera_z_slices)
        assert len(camera_eye_slices) == len(camera_y_slices)
        assert len(camera_eye_slices) == len(camera_x_slices)
        assert len(camera_eye_slices) == len(camera_dis_slices)
        assert len(camera_eye_slices) == len(camera_pos_slices)
        assert len(camera_eye_slices) == len(camera_rot_slices)
        assert len(camera_eye_slices) == len(camera_fov_slices)
        if not len(camera_eye_slices) == window:
            print(len(camera_eye_slices))
        out = {
            "camera_eye": camera_eye_slices,
            "camera_z": camera_z_slices,
            "camera_y": camera_y_slices,
            "camera_x": camera_x_slices,
            "Distance": camera_dis_slices,
            "Position": camera_pos_slices,
            "Rotation": camera_rot_slices,
            "Fov": camera_fov_slices
        }
        out_path = os.path.join(camera_c_out,file_name+'_slice'+str(slice_count)+'.json')
        with open(out_path, 'w') as of:
            json.dump(out, of)
        start_idx += stride_step
        slice_count += 1
    return slice_count

def SliceBoneMask(bone_mask_path, stride, length, max_num_slices, bone_mask_out):
    with open(bone_mask_path, 'r') as bmf:
        bone_mask_data = json.load(bmf)
    bone_mask = bone_mask_data["bone_mask"]
    total_frame = len(bone_mask)
    file_name = os.path.splitext(os.path.basename(bone_mask_path))[0]
    
    start_idx = 0
    window = int(length*fps)
    stride_step = int(stride*fps)
    slice_count = 0
    
    while start_idx <= total_frame - window  and slice_count < max_num_slices:
        bone_mask_slices = bone_mask[start_idx : start_idx + window]
        out = {'bone_mask':bone_mask_slices}
        out_path = os.path.join(bone_mask_out,file_name+'_slice'+str(slice_count)+'.json')
        with open(out_path, 'w') as of:
            json.dump(out, of)
        start_idx += stride_step
        slice_count += 1
    return slice_count

def SliceMotion_GlobalTransform2Keypoints(motion_gt_path, stride, length, max_num_slices, motion_keypoints_out):
    # print(motion_gt_path)
    with open(motion_gt_path, 'r', encoding="utf-8") as mf:
        motion_data = json.load(mf)
    motion_gt = motion_data["BoneKeyFrameTransformRecord"]
    total_frame = motion_data["BoneKeyFrameNumber"]
    motion_gt.sort(key=FrameTimeFunc)
    file_name = os.path.splitext(os.path.basename(motion_gt_path))[0][:-3]
    
    start_idx = 0
    window = int(length*fps)
    stride_step = int(stride*fps)
    slice_count = 0
    
    while start_idx <= total_frame - window  and slice_count < max_num_slices:
        keypoints_slices = []
        for i in range(window):
            # print(start_idx+i)
            keypoints_slices.append(GlobalTransform2Keypoints(motion_gt[start_idx+i]))
        out = {'Keypoints3D':keypoints_slices}
        out_path = os.path.join(motion_keypoints_out,file_name+'_kps3D_slice'+str(slice_count)+'.json')
        with open(out_path, 'w') as of:
            json.dump(out, of)
        start_idx += stride_step
        slice_count += 1
    return slice_count

def GlobalTransform2Keypoints_File(motion_gt_dir, motion_keypoints_out_dir):
    # print(motion_gt_path)
    motion_gt_list = glob.glob(os.path.join(motion_gt_dir,'*.json'))
    for motion_gt_path in motion_gt_list:
        with open(motion_gt_path, 'r', encoding="utf-8") as mf:
            motion_data = json.load(mf)
        motion_gt = motion_data["BoneKeyFrameTransformRecord"]
        total_frame = motion_data["BoneKeyFrameNumber"]
        motion_gt.sort(key=FrameTimeFunc)
        file_name = os.path.splitext(os.path.basename(motion_gt_path))[0][:-3]

        keypoints_slices = []
        for i in range(total_frame):
            keypoints_slices.append(GlobalTransform2Keypoints(motion_gt[i]))
        out = {'Keypoints3D':keypoints_slices}
        out_path = os.path.join(motion_keypoints_out_dir,file_name+'_kps3D.json')
        with open(out_path, 'w') as of:
            json.dump(out, of)