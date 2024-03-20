# align the length of audio, motion, and camera
import os
import json
import argparse
from tqdm import tqdm
import librosa as lr
import numpy as np
import soundfile as sf


fps = 30
def FrameTimeFunc(e):
    return e["FrameTime"]

def Align_AudioMotionCamera(output_dir, raw_dir, camera_json_dir, motion_transform_json_dir):
    output_audio_dir = os.path.join(output_dir,'Audio')
    output_motion_gt_dir = os.path.join(output_dir,'Simplified_MotionGlobalTransform')
    output_camera_kf_dir = os.path.join(output_dir,'CameraKeyframe')
    if not os.path.exists(output_audio_dir):
        os.mkdir(output_audio_dir)
    if not os.path.exists(output_motion_gt_dir):
        os.mkdir(output_motion_gt_dir)
    if not os.path.exists(output_camera_kf_dir):
        os.mkdir(output_camera_kf_dir)
    raw_list = os.listdir(raw_dir)
    for item in tqdm(raw_list):
        if not item.startswith('amc'):
            continue
        audio_path = os.path.join(raw_dir, item,'a'+item[3:]+'.wav')
        motion_gt_path = os.path.join(motion_transform_json_dir,'m'+item[3:]+'_GlobalTransform.json')
        camera_kf_path = os.path.join(camera_json_dir,'c'+item[3:]+'.json')

        audio, sr = lr.load(audio_path, sr=None)
        with open(motion_gt_path, 'r') as mf:
            motion_data = json.load(mf)
        with open(camera_kf_path, 'r') as cf:
            camera_data = json.load(cf)
        min_len = int(float(len(audio))/float(sr) *float(fps))
        if min_len > motion_data["BoneKeyFrameNumber"]:#But indeed here is not all keyframes but interpolated frames in our Simplified_MotionGlobalTransform files
            min_len = motion_data["BoneKeyFrameNumber"]
        camera_keyframes = camera_data["CameraKeyFrameRecord"]
        camera_keyframes.sort(key=FrameTimeFunc)
        for c_i in range(1,len(camera_keyframes)+1):
            if camera_keyframes[-c_i]['FrameTime'] >= min_len:
                continue
            else:
                min_len = camera_keyframes[-c_i]['FrameTime'] + 1
                break
        #write audio
        if int(float(min_len)/float(fps)*float(sr)) >= len(audio):
            sf.write(f"{output_audio_dir}/a{item[3:]}.wav", audio, sr)
        else:
            sf.write(f"{output_audio_dir}/a{item[3:]}.wav", audio[:int(float(min_len)/float(fps)*float(sr))+1], sr)
        #write motion
        motion_data["BoneKeyFrameTransformRecord"] = motion_data["BoneKeyFrameTransformRecord"][:min_len]
        motion_data["BoneKeyFrameNumber"] = min_len
        with open(f"{output_motion_gt_dir}/m{item[3:]}_gt.json", 'w') as mof:
            json.dump(motion_data, 
                        mof,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False)
        #write camera
        camera_data["CameraKeyFrameRecord"] = []
        for c_kf in camera_keyframes:
            if c_kf['FrameTime'] >= min_len:
                continue
            camera_data["CameraKeyFrameRecord"].append(c_kf)
        with open(f"{output_camera_kf_dir}/c{item[3:]}.json", 'w') as cof:
            json.dump(camera_data, 
                        cof,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False)

parser = argparse.ArgumentParser()
parser.add_argument('--amc_raw_dir', type=str, default='../../data/wangzx_DCM/amc_raw_data/')
parser.add_argument('--amc_raw_camera_json_dir', type=str, default='../../data/wangzx_DCM/data/amc_camera_json/')
parser.add_argument('--amc_motion_transform_json_dir', type=str, default='../../data/wangzx_DCM/data/Simplified_MotionGlobalTransform/')
parser.add_argument('--amc_aligned_dir', type=str, default='DCM_data/amc_aligned_data/')
args = parser.parse_args()

if __name__ == '__main__':
    amc_aligned_dir = args.amc_aligned_dir
    if not os.path.exists(amc_aligned_dir):
        os.mkdir(amc_aligned_dir)
    Align_AudioMotionCamera(amc_aligned_dir, args.amc_raw_dir, args.amc_raw_camera_json_dir, args.amc_motion_transform_json_dir)