import os
import json
from pathlib import Path
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
sys.path.append(".")
from my_utils.camera_postprocess import camera_centric2vmd_dir, camera_centric2vmd_json
import math

def ExtendCameraResult(source_camera_json, target_camera_extend_json, split_json, split_data=None):
    if split_data==None:
        with open(split_json, 'r') as sf:
            split_data = json.load(sf)

    Path(os.path.dirname(target_camera_extend_json)).mkdir(parents=True, exist_ok=True)

    camera_json_name = os.path.basename(source_camera_json)
    #camera_json_name is like 'c{data number}_{split number}_{slice number/test}.json' or 'c{data number}_{slice number/test}.json' for not split data
    camera_json_name_split = camera_json_name.split('_')
    camera_index = camera_json_name_split[0][1:]
    if len(camera_json_name_split) == 2:#no split

        with open(source_camera_json, 'r') as cf:
            camera_data = json.load(cf)
        camera_eye = np.array(camera_data['camera_eye'])
        camera_z = np.array(camera_data['camera_z'])
        camera_y = np.array(camera_data['camera_y'])
        camera_x = np.array(camera_data['camera_x'])
        camera_rot = np.array(camera_data['camera_rot'])
        camera_fov = np.array(camera_data['Fov'])

    elif len(camera_json_name_split) == 3:
        split_num = int(camera_json_name_split[1])
        assert camera_index in split_data
        start_frame = split_data[camera_index][split_num][0]
        end_frame = split_data[camera_index][split_num][1]
        
        with open(source_camera_json, 'r') as cf:
            camera_data = json.load(cf)
        camera_eye = np.array(camera_data['camera_eye'])
        camera_z = np.array(camera_data['camera_z'])
        camera_y = np.array(camera_data['camera_y'])
        camera_x = np.array(camera_data['camera_x'])
        camera_rot = np.array(camera_data['camera_rot'])
        camera_fov = np.array(camera_data['Fov'])

        camera_z = np.concatenate((np.zeros((start_frame,camera_z.shape[1]),dtype= np.float64)+camera_z[0],camera_z))
        camera_y = np.concatenate((np.zeros((start_frame,camera_y.shape[1]),dtype= np.float64)+camera_y[0],camera_y))
        camera_x = np.concatenate((np.zeros((start_frame,camera_x.shape[1]),dtype= np.float64)+camera_x[0],camera_x))
        camera_rot = np.concatenate((np.zeros((start_frame,camera_rot.shape[1]),dtype= np.float64)+camera_rot[0],camera_rot))
        camera_eye = np.concatenate((np.zeros((start_frame,camera_eye.shape[1]),dtype= np.float64)+camera_eye[0],camera_eye))
        camera_fov = np.concatenate((np.zeros((start_frame,camera_fov.shape[1]),dtype= np.float64)+camera_fov[0],camera_fov))

    with open(target_camera_extend_json, 'w') as ocf:
        json.dump({
            'FrameTime':len(camera_eye),
            'camera_eye':camera_eye.tolist(),
            'camera_z':camera_z.tolist(),
            'camera_y':camera_y.tolist(),
            'camera_x':camera_x.tolist(),
            'camera_rot':camera_rot.tolist(),
            'Fov':camera_fov.tolist()
        },ocf)

def ExtendCameraResultDir(source_camera_dir, target_camera_extend_dir, split_json):
    Path(target_camera_extend_dir).mkdir(parents=True, exist_ok=True)
    camera_list = os.listdir(source_camera_dir)
    with open(split_json, 'r') as sf:
        split_data = json.load(sf)
    
    
    for camera_json in tqdm(camera_list):
        ExtendCameraResult(os.path.join(args.source_camera_dir,camera_json), os.path.join(args.target_camera_extend_dir,camera_json), split_json, split_data)
        
parser = argparse.ArgumentParser()
parser.add_argument('--split_json', type=str, default='DCM_data/split/long2short.json', help='json for split information')
parser.add_argument('--source_camera_dir', type=str, default='', help='source camera json directory')
parser.add_argument('--target_camera_extend_dir', type=str, default='', help='target directory of extend camera json')
parser.add_argument('--target_camera_vmdjson_dir', type=str, default='', help='target directory of extend camera json in vmd format')

parser.add_argument('--source_camera_json', type=str, default='', help='source camera json')
parser.add_argument('--target_camera_extend_json', type=str, default='', help='target path of extend camera json')
parser.add_argument('--target_camera_vmdjson', type=str, default='', help='target path of extend camera json in vmd format')
args = parser.parse_args()

if __name__ == "__main__":
    if args.source_camera_dir == '':
        if args.source_camera_json == '':
            print('Please enter source json or directory')
        else:
            ExtendCameraResult(args.source_camera_json, args.target_camera_extend_json, args.split_json)
            camera_centric2vmd_json(args.target_camera_extend_json, args.target_camera_vmdjson)
    else:
        ExtendCameraResultDir(args.source_camera_dir, args.target_camera_extend_dir, args.split_json)
        camera_centric2vmd_dir(args.target_camera_extend_dir, args.target_camera_vmdjson_dir)