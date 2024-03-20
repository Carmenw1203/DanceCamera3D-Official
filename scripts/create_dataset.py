import os
import argparse
import math
from pathlib import Path
import sys 
sys.path.append("..")
sys.path.append(".")
from my_utils.slice import *
from tqdm import tqdm
from audio_extraction.jukebox_features import extract_folder as jukebox_extract

def GetIndex(e):
    e_name = e.split('.')[0]
    e_name_split = e_name[1:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def GetIndexBM(e):
    e_name = e.split('.')[0]
    e_name_split = e_name[2:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def SliceAmc(audio_dir,camera_c_dir,motion_gt_dir, bone_mask_dir, data_dir, stride=0.5, length=5):
    audio_out = audio_dir+'_sliced'
    camera_c_out = camera_c_dir+'_sliced'
    bone_mask_out = bone_mask_dir + '_sliced'
    motion_gt_out = motion_gt_dir+'_sliced'
    motion_keypoints_out = data_dir+'/Keypoints3D_sliced'
    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(camera_c_out, exist_ok=True)
    os.makedirs(bone_mask_out, exist_ok=True)
    os.makedirs(motion_gt_out, exist_ok=True)
    os.makedirs(motion_keypoints_out, exist_ok=True)
    audio_list = os.listdir(audio_dir)
    camera_c_list = os.listdir(camera_c_dir)
    bone_mask_list = os.listdir(bone_mask_dir)
    motion_gt_list = os.listdir(motion_gt_dir)
    audio_list.sort(key=GetIndex)
    camera_c_list.sort(key=GetIndex)
    bone_mask_list.sort(key=GetIndexBM)
    motion_gt_list.sort(key=GetIndex)
    assert len(audio_list) == len(camera_c_list)
    assert len(audio_list) == len(motion_gt_list)
    assert len(audio_list) == len(bone_mask_list)

    for audio_i, camera_c_i, motion_gt_i, bone_mask_i in tqdm(zip(audio_list, camera_c_list, motion_gt_list, bone_mask_list)):
        audio_path = os.path.join(audio_dir,audio_i)
        camera_c_path = os.path.join(camera_c_dir,camera_c_i)
        bone_mask_path = os.path.join(bone_mask_dir,bone_mask_i)
        motion_gt_path = os.path.join(motion_gt_dir,motion_gt_i)
        # sclice camera first because camera is not longer than audio and motion
        camera_c_slices = SliceCameraCentric(camera_c_path, stride, length, camera_c_out)
        bone_mask_slices = SliceBoneMask(bone_mask_path, stride, length, camera_c_slices, bone_mask_out)
        motion_keypoints_slices = SliceMotion_GlobalTransform2Keypoints(motion_gt_path, stride, length, camera_c_slices, motion_keypoints_out)
        
        audio_slices = SliceAudio(audio_path, stride, length, camera_c_slices, audio_out)
        assert motion_keypoints_slices == camera_c_slices
        assert motion_keypoints_slices == audio_slices
        assert motion_keypoints_slices == bone_mask_slices
        
        
def DetactBoneMask(camera_c_dir, motion_gt_dir, bone_mask_dir):
    os.makedirs(bone_mask_dir, exist_ok=True)
    camera_c_list = os.listdir(camera_c_dir)
    motion_gt_list = os.listdir(motion_gt_dir)
    camera_c_list.sort(key=GetIndex)
    motion_gt_list.sort(key=GetIndex)
    assert len(motion_gt_list) == len(camera_c_list)
    for camera_c_i, motion_gt_i in tqdm(zip(camera_c_list, motion_gt_list)):
        camera_c_path = os.path.join(camera_c_dir,camera_c_i)
        motion_gt_path = os.path.join(motion_gt_dir,motion_gt_i)
        with open(camera_c_path, 'r') as cf:
            camera_data = json.load(cf)
        with open(motion_gt_path, 'r') as mf:
            motion_data = json.load(mf)
        data_len = min(len(camera_data["Fov"]),motion_data["BoneKeyFrameNumber"])
        camera_eye = np.array(camera_data["camera_eye"],dtype='float32')[:data_len]
        camera_z = np.array(camera_data["camera_z"],dtype='float32')[:data_len]
        camera_y = np.array(camera_data["camera_y"],dtype='float32')[:data_len]
        camera_x = np.array(camera_data["camera_x"],dtype='float32')[:data_len]
        camera_fov = np.array(camera_data["Fov"],dtype='float32')[:data_len]
        keypoints = []
        for gt_i in motion_data["BoneKeyFrameTransformRecord"]:
            keypoints.append(GlobalTransform2Keypoints(gt_i))
        keypoints = np.array(keypoints,dtype='float32')[:data_len]#(t,60*3)
        keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)#(t,60,3)
        keypoints = keypoints.transpose(1,0,2)#(60,t,3)
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

def CreateDataset(args):
    bone_mask_dir = os.path.join(args.data_dir, "BoneMask")
    DetactBoneMask(args.camera_c_dir,args.motion_gt_dir,bone_mask_dir)
    #slice audio camera and motions into sliding windows to create dataset
    SliceAmc(args.audio_dir,args.camera_c_dir,args.motion_gt_dir,bone_mask_dir, args.data_dir)
    # extract acoustic features
    if args.extract_jukebox:
        jukebox_extract(args.audio_dir+'_sliced',args.data_dir+'/jukebox_feats')
    
    

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, default='DCM_data/amc_aligned_data/Audio')
parser.add_argument('--camera_c_dir', type=str, default='DCM_data/amc_aligned_data/CameraCentric')
parser.add_argument('--motion_gt_dir', type=str, default='DCM_data/amc_aligned_data/Simplified_MotionGlobalTransform')
parser.add_argument('--data_dir', type=str, default='DCM_data/amc_aligned_data')
parser.add_argument('--split_record', type=str, default='DCM_data/split/long2short.json')
parser.add_argument('--output_dir', type=str, default='DCM_data/amc_aligned_data_split/')
parser.add_argument('--extract_jukebox', action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    CreateDataset(args)