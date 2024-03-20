from my_utils.slice import GlobalTransform2Keypoints
import os
import math
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

def GetIndex(e):
    
    e_name = e.split(os.sep)[-1].split('.')[0]
    # print(e_name)
    e_name_split = e_name[1:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 3:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def DetactBoneMask(camera_c_dir, motion_gt_dir, bone_mask_dir):
    os.makedirs(bone_mask_dir, exist_ok=True)
    camera_c_list = os.listdir(camera_c_dir)
    motion_gt_list = os.listdir(motion_gt_dir)
    camera_c_list.sort(key=GetIndex)
    motion_gt_list.sort(key=GetIndex)
    assert len(motion_gt_list) == len(camera_c_list)
    #debug
    for camera_c_i, motion_gt_i in tqdm(zip(camera_c_list, motion_gt_list)):
        camera_c_path = os.path.join(camera_c_dir,camera_c_i)
        motion_gt_path = os.path.join(motion_gt_dir,motion_gt_i)
        print(camera_c_i, motion_gt_i)
        with open(camera_c_path, 'r') as cf:
            camera_data = json.load(cf)
        with open(motion_gt_path, 'r') as mf:
            motion_data = json.load(mf)
        data_len = min(len(camera_data["Fov"]),motion_data["BoneKeyFrameNumber"])
        camera_eye = np.array(camera_data["camera_eye"],dtype='float32')[:data_len]
        camera_z = np.array(camera_data["camera_z"],dtype='float32')[:data_len]# has been normalized
        camera_y = np.array(camera_data["camera_y"],dtype='float32')[:data_len]
        camera_x = np.array(camera_data["camera_x"],dtype='float32')[:data_len]
        camera_fov = np.array(camera_data["Fov"],dtype='float32')[:data_len].reshape(-1)
        keypoints = []
        for gt_i in motion_data["BoneKeyFrameTransformRecord"]:
            keypoints.append(GlobalTransform2Keypoints(gt_i))
        keypoints = np.array(keypoints,dtype='float32')[:data_len]#(t,61*3)
        keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)#(t,61,3)
        keypoints = keypoints.transpose(1,0,2)#(61,t,3)
        key2eye = keypoints - camera_eye
        kps_yz = key2eye - camera_x* np.sum(key2eye*camera_x, axis=-1, keepdims = True)
        kps_xz = key2eye - camera_y* np.sum(key2eye*camera_y, axis=-1, keepdims = True) 
        cos_y_z = np.sum(kps_yz*camera_z, axis=-1)
        cos_x_z = np.sum(kps_xz*camera_z, axis=-1)
        cos_fov = np.cos(camera_fov*0.5/180 * math.pi)

        bone_mask = (cos_y_z >= cos_fov*np.sqrt(np.sum(kps_yz*kps_yz, axis=-1))).astype(np.int) + (cos_x_z >= cos_fov*np.sqrt(np.sum(kps_xz*kps_xz, axis=-1))).astype(np.int)
        bone_mask = (bone_mask >= 2).astype(np.int32).transpose(1,0)#
        bone_mask_name = 'bm'+camera_c_i[1:]
        bone_mask_path = os.path.join(bone_mask_dir,bone_mask_name)
        with open(bone_mask_path, 'w') as bf:
            json.dump({
                "bone_mask":bone_mask.tolist()
            },bf)
        # break