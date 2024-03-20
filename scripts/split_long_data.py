# for the reason that raw data varies in time for a vary large variance, 
# in this file, we decide to split long data pieces into small pieces 
# so that the data variance will decrease and we could divide the dataset more easily
# in order to keep the completeness of camera movement we split the data at the keyframe times with randomness
import os
import json
import shutil
import random
import argparse
from tqdm import tqdm
import librosa as lr
import numpy as np
import soundfile as sf
from pathlib import Path
import math

fps = 30
lower_bound = 17
upper_bound = 35

def FrameTimeFunc(e):
    return e["FrameTime"]

def LongData2SplitRecord(camera_dir,split_json):
    camera_files = os.listdir(camera_dir)
    split_dict = {}
    for camera_file in tqdm(camera_files):
        print(camera_file)
        cur_index = camera_file[1:-5]
        split_dict[cur_index] = []
        camera_path = os.path.join(camera_dir, camera_file)
        with open(camera_path, 'r') as cf:
            camera_data = json.load(cf)
        keyframes = camera_data["CameraKeyFrameRecord"]
        keyframes.sort(key=FrameTimeFunc)
        # no split on short sequences
        if keyframes[-1]['FrameTime'] <= upper_bound * fps:
            continue
        # loop for long sequences split
        while(keyframes[-1]['FrameTime'] - keyframes[0]['FrameTime'] > upper_bound*fps):
            random_index = int(random.random()*len(keyframes))
            if keyframes[random_index]['FrameTime'] - keyframes[0]['FrameTime'] > upper_bound*fps or keyframes[random_index]['FrameTime'] - keyframes[0]['FrameTime'] < lower_bound*fps:
                continue
            if keyframes[-1]['FrameTime'] - keyframes[random_index]['FrameTime'] < lower_bound*fps:
                continue
            split_dict[cur_index].append((keyframes[0]['FrameTime'],keyframes[random_index]['FrameTime']))
            keyframes = keyframes[random_index:]
        split_dict[cur_index].append((keyframes[0]['FrameTime'],keyframes[-1]['FrameTime']))
    with open(split_json,'w') as sf:
        json.dump(split_dict, sf)
    return split_dict
def ReadSplitRecord(split_json):
    with open(split_json,'r') as sf:
        split_dict = json.load(sf)
    return split_dict      
def SplitLongDataFromRecord(args, split_dict):
    audio_files = os.listdir(args.audio_dir)
    camera_c_files = os.listdir(args.camera_c_dir)
    motion_files = os.listdir(args.motion_dir)
    out_audio_dir = os.path.join(args.output_dir,'Audio')
    out_camera_c_dir = os.path.join(args.output_dir, 'CameraCentric')
    out_motion_dir = os.path.join(args.output_dir, 'Simplified_MotionGlobalTransform')
    Path(out_audio_dir).mkdir(parents=True, exist_ok=True)
    Path(out_camera_c_dir).mkdir(parents=True, exist_ok=True)
    Path(out_motion_dir).mkdir(parents=True, exist_ok=True)
    for audio_file in tqdm(audio_files):
        index = audio_file[1:-4]
        camera_c_file = 'c'+index+'.json'
        motion_file = 'm'+index+'_gt.json'
        
        audio_path = os.path.join(args.audio_dir, audio_file)
        camera_c_path = os.path.join(args.camera_c_dir, camera_c_file)
        motion_path = os.path.join(args.motion_dir, motion_file)
        if len(split_dict[index])==0:
            
            shutil.copyfile(audio_path, f"{out_audio_dir}/{audio_file}")
            shutil.copyfile(camera_c_path, f"{out_camera_c_dir}/{camera_c_file}")
            shutil.copyfile(motion_path, f"{out_motion_dir}/{motion_file}")
            continue

        audio, sr = lr.load(audio_path, sr=None)
        with open(camera_c_path, 'r') as ccf:
            camera_c_data = json.load(ccf)
        with open(motion_path, 'r') as mf:
            motion_data = json.load(mf)
        tmp_cnt = 0
        for [start_frame,end_frame] in split_dict[index]:
            if math.ceil(float(end_frame+1)/float(fps)*float(sr)) >= len(audio):
                tmp_audio = audio[int(float(start_frame)/float(fps)*float(sr)):]
            else:
                tmp_audio = audio[int(float(start_frame)/float(fps)*float(sr)):math.ceil(float(end_frame+1)/float(fps)*float(sr))]
            tmp_motion = motion_data.copy()
            tmp_camera_c = camera_c_data.copy()
            tmp_motion["BoneKeyFrameNumber"] = end_frame - start_frame + 1
            tmp_motion["BoneKeyFrameTransformRecord"] = motion_data["BoneKeyFrameTransformRecord"][start_frame:end_frame+1]
            
            for camera_args in tmp_camera_c:
                if camera_args == 'FrameTime':
                    tmp_camera_c[camera_args] = end_frame - start_frame + 1
                else:
                    tmp_camera_c[camera_args] = tmp_camera_c[camera_args][start_frame:end_frame+1]

            sf.write(f"{out_audio_dir}/a{index}_{str(tmp_cnt)}.wav", tmp_audio, sr)
            with open(f"{out_camera_c_dir}/c{index}_{str(tmp_cnt)}.json", 'w') as ccof:
                json.dump(tmp_camera_c,
                        ccof,
                        indent=2,  
                        sort_keys=True,  
                        ensure_ascii=False)
            with open(f"{out_motion_dir}/m{index}_{str(tmp_cnt)}_gt.json", 'w') as mof:
                json.dump(tmp_motion,
                        mof,
                        indent=2,  
                        sort_keys=True,  
                        ensure_ascii=False)
            tmp_cnt += 1
        
parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, default='DCM_data/amc_aligned_data/Audio')
parser.add_argument('--camera_kf_dir', type=str, default='DCM_data/amc_aligned_data/CameraKeyframe')
parser.add_argument('--camera_c_dir', type=str, default='DCM_data/amc_aligned_data/CameraCentric')
parser.add_argument('--motion_dir', type=str, default='DCM_data/amc_aligned_data/Simplified_MotionGlobalTransform/')
parser.add_argument('--split_record', type=str, default='DCM_data/split/long2short.json')
parser.add_argument('--output_dir', type=str, default='DCM_data/amc_aligned_data_split/')
parser.add_argument('--split_from_file', type=bool, default=True)
args = parser.parse_args()


if __name__ == '__main__':
    
    # from record file
    if args.split_from_file:
        split_record = ReadSplitRecord(args.split_record)
        SplitLongDataFromRecord(args, split_record)
    else:
        # from calculation
        split_record = LongData2SplitRecord(args.camera_kf_dir,args.split_record)
