import glob
import os
import sys
sys.path.append("..")
sys.path.append(".")
from tqdm import tqdm
import json
import torch
import random
import jukemirlib
from pathlib import Path
import numpy as np
from args import parse_test_opt
from my_utils.slice import *
from tempfile import TemporaryDirectory
from DanceCamera3D import DanceCamera3D 
from audio_extraction.jukebox_features import extract_folder as jukebox_extract

def GetSliceNumber(e):
    e_name = e.split('/')[-1].split('.')[0]
    e_name_split = e_name[1:].split('_slice')
    return int(e_name_split[1])

def GetNameIndexSlice(e):
    e_name = e.split('/')[-1].split('.')[0]
    e_name_split = e_name[1:].split('_slice')
    m_index = e_name_split[0]
    m_index_split = m_index.split('_')
    return m_index_split[0]+'_'+m_index_split[1]+'_'+e_name_split[1]

def GetNameIndex(e):
    
    e_name = e.split(os.sep)[-1].split('.')[0]
    e_name_split = e_name[1:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index      

def test_long(opt):

    all_m_cond = []
    all_p_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir_i in dir_list:
            m_cond_list = []
            p_cond_list = []
            music_dir_i = os.path.join(dir_i, 'Audio_LongSliced')
            feature_dir_i = os.path.join(dir_i, 'jukebox_feats_LongSliced')
            pose_dir_i = os.path.join(dir_i, 'Keypoints3D_LongSliced')
            music_stitch_dir_i = os.path.join(dir_i, 'Audio_Stitch')
            Path(music_stitch_dir_i).mkdir(parents=True, exist_ok=True)
            music_slices_list = sorted(glob.glob(f"{music_dir_i}/*.wav"), key=GetSliceNumber)
            feature_slices_list = sorted(glob.glob(f"{feature_dir_i}/*.npy"), key=GetSliceNumber)
            pose_slices_list = sorted(glob.glob(f"{pose_dir_i}/*.json"), key=GetSliceNumber)
            print(music_dir_i, feature_dir_i, pose_dir_i)
            assert len(music_slices_list) == len(feature_slices_list) == len(pose_slices_list)
            for pose_slice, feature_slice in zip(pose_slices_list, feature_slices_list):
                m_cond_list.append(np.array(np.load(feature_slice),dtype='float32'))
                with open(pose_slice, 'r') as psf:
                    pose_slice_data = json.load(psf)
                p_cond_list.append(np.array(pose_slice_data["Keypoints3D"],dtype='float32'))
            all_filenames.append(music_slices_list)
            all_m_cond.append(m_cond_list)
            all_p_cond.append(p_cond_list)
    else:
        print("Computing features for input music")
        music_list = sorted(glob.glob(os.path.join(opt.music_dir, "*.wav")), key=GetNameIndex)
        pose_list = sorted(glob.glob(os.path.join(opt.motionGT_dir, "*.json")), key=GetNameIndex)
        camera_list = sorted(glob.glob(os.path.join(opt.camera_dir, "*.json")), key=GetNameIndex)
        assert len(music_list) == len(pose_list) == len(camera_list) 
        cnt = 0
        for music_file, pose_file, camera_file in zip(music_list, pose_list, camera_list):
            # create folder to cache the feature (or use the cache folder if specified)
            songname = os.path.splitext(os.path.basename(music_file))[0]
            save_dir = os.path.join(opt.feature_cache_dir, songname)
            save_music_dir = os.path.join(save_dir, 'Audio_LongSliced')
            save_feature_dir = os.path.join(save_dir, 'jukebox_feats_LongSliced')
            save_pose_dir = os.path.join(save_dir, 'Keypoints3D_LongSliced')
            save_camera_dir = os.path.join(save_dir, 'CameraCentric_LongSliced')
            save_music_stitch_dir_i = os.path.join(save_dir, 'Audio_Stitch')
            Path(save_music_dir).mkdir(parents=True, exist_ok=True)
            Path(save_pose_dir).mkdir(parents=True, exist_ok=True)
            Path(save_feature_dir).mkdir(parents=True, exist_ok=True)
            Path(save_camera_dir).mkdir(parents=True, exist_ok=True)
            Path(save_music_stitch_dir_i).mkdir(parents=True, exist_ok=True)
            # slice the audio file
            print(f"Slicing {music_file} and {pose_file}")
            camera_slices_num = SliceCameraCentric(camera_file, 2.5, 5.0, save_camera_dir)
            pose_slices_num = SliceMotion_GlobalTransform2Keypoints(pose_file, 2.5, 5.0, camera_slices_num, save_pose_dir)
            music_slices_num = SliceAudio(music_file, 2.5, 5.0, pose_slices_num, save_music_dir)
            
            assert music_slices_num == pose_slices_num == camera_slices_num

            # generate juke representations
            print(f"Computing features for {music_file}")
            jukebox_extract(save_music_dir,save_feature_dir)

            music_slices_list = sorted(glob.glob(f"{save_music_dir}/*.wav"), key=GetSliceNumber)
            pose_slices_list = sorted(glob.glob(f"{save_pose_dir}/*.json"), key=GetSliceNumber)
            feature_slices_list = sorted(glob.glob(f"{save_feature_dir}/*.npy"), key=GetSliceNumber)

            m_cond_list = []
            p_cond_list = []
            
            for pose_slice, feature_slice in zip(pose_slices_list, feature_slices_list):
                m_cond_list.append(np.array(np.load(feature_slice),dtype='float32'))
                with open(pose_slice, 'r') as psf:
                    pose_slice_data = json.load(psf)
                p_cond_list.append(np.array(pose_slice_data["Keypoints3D"],dtype='float32'))

            all_m_cond.append(m_cond_list)
            all_p_cond.append(p_cond_list)
            all_filenames.append(music_slices_list)
            
    if(opt.backbone == 'diffusion'):
        model = DanceCamera3D(feature_type = opt.feature_type,
                            checkpoint_path = opt.checkpoint,
                            camera_format = opt.camera_format,
                            condition_separation_CFG = opt.condition_separation_CFG,
                            gw1 = opt.gw1,
                            gw2 = opt.gw2)
    model.eval()

    print("Generating dances")
    for i in range(len(all_m_cond)):

        pose_cond = torch.from_numpy(np.array(all_p_cond[i]))
        music_cond = torch.from_numpy(np.array(all_m_cond[i]))
        
        model.render_sample(
            pose_cond, music_cond, all_filenames[i], opt.label, opt.render_dir, render_count=-1, render_videos=opt.render_videos, test_mode = opt.test_mode
        )
    print("Done")

if __name__ == "__main__":
    opt = parse_test_opt()
    test_long(opt)