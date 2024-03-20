import os
import json
import sys
sys.path.append("..")
sys.path.append(".")
import argparse
from tqdm import tqdm
from model.VmdBezier import VmdBezier
from my_utils.curve import Mmd_Curve
import glm
import math
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--camera_centric_json_dir', type=str, default='output/camera_centric_json')
parser.add_argument('--camera_json_dir', type=str, default='output/camera_json')
parser.add_argument('--camera_vmd_json_dir', type=str, default='output/camera_vmd_json_results')

def camera_centric2vmd_json(input_json, output_json):
    with open(input_json, 'r') as fin:
            camera_data_centric = json.load(fin)
    max_frame = camera_data_centric["FrameTime"]
    camera_eye = camera_data_centric["camera_eye"]#camera_position
    camera_z = camera_data_centric["camera_z"]#camera_facing_direction
    camera_y = camera_data_centric["camera_y"]#camera_up_direction
    camera_x = camera_data_centric["camera_x"]#camera_horizontal_direction
    camera_rot = camera_data_centric["camera_rot"]
    camera_fov = np.array(camera_data_centric["Fov"]).reshape(-1).tolist()#camera field of view
    
    out_vmd_camera = {}
    out_vmd_camera["CameraKeyFrameNumber"] = max_frame
    out_vmd_camera["CameraKeyFrameRecord"] = []

    
    for i in tqdm(range(max_frame)):
        out_frame_record = {}
        out_frame_record['Curve'] = Mmd_Curve.default_curve
        out_frame_record['Distance'] = 0.0
        out_frame_record['FrameTime'] = i
        out_frame_record['Orthographic'] = 0
        out_frame_record['Position'] = {}
        out_frame_record['Position']['x'] = camera_eye[i][0]
        out_frame_record['Position']['y'] = camera_eye[i][1]
        out_frame_record['Position']['z'] = camera_eye[i][2]*(-1)
        out_frame_record['Rotation'] = {}
        out_frame_record['ViewAngle'] = camera_fov[i]
        out_frame_record['Rotation']['z'] = camera_rot[i][2]
        out_frame_record['Rotation']['y'] = camera_rot[i][1]
        out_frame_record['Rotation']['x'] = camera_rot[i][0]
        out_vmd_camera["CameraKeyFrameRecord"].append(out_frame_record)
        
    Path(os.path.dirname(output_json)).mkdir(parents=True, exist_ok=True)   
    with open(output_json,'w') as fout:
        json.dump(out_vmd_camera,fout)
def camera_centric2vmd_dir(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    input_jsons = os.listdir(input_dir)
    for i_json in input_jsons:
        i_path = os.path.join(input_dir, i_json)
        o_path = os.path.join(output_dir, i_json)
        camera_centric2vmd_json(i_path,o_path)

def camera2vmd(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    input_jsons = os.listdir(input_dir)
    for i_json in input_jsons:
        i_path = os.path.join(input_dir, i_json)
        o_path = os.path.join(output_dir, i_json)
        with open(i_path, 'r') as fin:
            camera_data = json.load(fin)
        max_frame = camera_data["FrameTime"]
        camera_pos = camera_data["camera_pos"]
        camera_rot = camera_data["camera_rot"]
        camera_dis = camera_data["camera_dis"]
        camera_fov = camera_data["Fov"]#camera field of view
        
        out_vmd_camera = {}
        out_vmd_camera["CameraKeyFrameNumber"] = max_frame
        out_vmd_camera["CameraKeyFrameRecord"] = []

        
        for i in tqdm(range(max_frame)):
            out_frame_record = {}
            out_frame_record['Curve'] = Mmd_Curve.default_curve
            out_frame_record['Distance'] = camera_dis[i][0]
            out_frame_record['FrameTime'] = i
            out_frame_record['Orthographic'] = 0
            out_frame_record['Position'] = {}
            out_frame_record['Position']['x'] = float(camera_pos[i][0])
            out_frame_record['Position']['y'] = float(camera_pos[i][1])
            out_frame_record['Position']['z'] = float(camera_pos[i][2])
            out_frame_record['Rotation'] = {}
            out_frame_record['ViewAngle'] = float(camera_fov[i][0])

            out_frame_record['Rotation']['z'] = float(camera_rot[i][2])
            out_frame_record['Rotation']['y'] = float(camera_rot[i][1])
            out_frame_record['Rotation']['x'] = float(camera_rot[i][0])
            out_vmd_camera["CameraKeyFrameRecord"].append(out_frame_record)
            
            
        with open(o_path,'w') as fout:
            json.dump(out_vmd_camera,fout)
        
if __name__ == '__main__':
    args = parser.parse_args()

    camera_centric_dir = args.camera_centric_json_dir
    camera_dir = args.camera_json_dir
    camera_vmd_dir = args.camera_vmd_json_dir

    # convert usable camera parameters--camera centric represention to vmd camera data--camera vmd represention
    # camera centric represention:
    # (camera_position--eye, 
    #  camera_facing_direction--camera_z, 
    #  camera_up_direction--camera_y, 
    #  camera_horizontal_direction--camera_x, 
    #  fov)

    # camera vmd represention:
    # (distance--from the interest, 
    # position/interest--rotated around this point,
    # rotation, 
    # veiw angle/fov)

    # camera_centric2vmd(camera_centric_dir, camera_vmd_dir)

    camera2vmd(camera_dir, camera_vmd_dir)
