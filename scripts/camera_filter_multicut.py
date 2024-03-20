# script for smoothing post-processing
import os
import sys
sys.path.append("..")
sys.path.append(".")
import json
import argparse
import numpy as np
from tqdm import tqdm
import glm
import math
from scipy import signal
from pathlib import Path
from my_utils.tv_denoise import denoising_1D_TV

def AdaptiveSFT(input_signal,input_keyframes):
    # print(input_keyframes)
    output_signal = np.zeros_like(input_signal, dtype=np.float64)
    # b, a = signal.butter(8, 0.05, 'lowpass')
    for i in range(output_signal.shape[1]):
        for j in range(len(input_keyframes)):
            if j == len(input_keyframes) - 1:
                wl = output_signal.shape[0] - input_keyframes[j]
                if wl % 2 == 0:
                    wl -= 1
                if wl > 75:
                    wl=75
                po = 2
                if po >= wl:
                    po = wl - 1
                output_signal[input_keyframes[j]:,i] = signal.savgol_filter(input_signal[input_keyframes[j]:,i], window_length=wl, polyorder=po)
            else:
                wl = input_keyframes[j+1] - input_keyframes[j]
                if wl % 2 == 0:
                    wl -= 1
                if wl > 75:
                    wl=75
                po = 2
                if po >= wl:
                    po = wl - 1
                # print(po)
                output_signal[input_keyframes[j]:input_keyframes[j+1],i] = signal.savgol_filter(input_signal[input_keyframes[j]:input_keyframes[j+1],i], window_length=wl, polyorder=po)
    return output_signal


def TV_Detect_Keyframes(input_signal):
    keyframes = np.array([0])
    changindex = np.array([])
    N = input_signal.shape[0]
    sigma = 0.5
    lamda = np.sqrt(sigma * N) / 5 *100
    for i in range(input_signal.shape[1]):
        X1 = denoising_1D_TV(input_signal[:,i], lamda / 2)
        X1_v = np.abs(X1[1:] - X1[:-1])
        itemindex = np.argwhere(X1_v[:-1] > 0)
        # print(itemindex)
        # print(keyframes.shape,itemindex.shape)
        changindex = np.concatenate([changindex, itemindex.reshape(-1)])
    changindex = sorted(np.unique(changindex))
    
    
    left_edge = changindex[0]
    right_edge = changindex[0]
    j = 1
    while(j< len(changindex)):
        
        if changindex[j] - right_edge < 30:
            right_edge = changindex[j]
            j += 1
        else:
            keyframes = np.append(keyframes,int((left_edge + right_edge)//2))
            left_edge = changindex[j]
            right_edge = changindex[j]
            j += 1
    # print(keyframes)
    return keyframes


def FilterCamera(args):
    camera_dir = args.input_dir
    camera_filtered_dir = args.output_dir

    Path(camera_filtered_dir).mkdir(parents=True, exist_ok=True)

    camera_list = os.listdir(camera_dir)
    for camera_file in tqdm(camera_list):
        with open(os.path.join(camera_dir,camera_file), 'r') as cf:
            camera_data = json.load(cf)

        camera_fov = np.array(camera_data['Fov'], dtype=np.float64)
        camera_eye = np.array(camera_data['camera_eye'], dtype=np.float64)
        camera_dis = np.zeros_like(camera_fov)
        camera_rot = np.array(camera_data['camera_rot'], dtype=np.float64)

        tmp_keyframes = TV_Detect_Keyframes(camera_eye)
        out_camera_fov = AdaptiveSFT(camera_fov,tmp_keyframes)
        out_camera_eye = AdaptiveSFT(camera_eye,tmp_keyframes)
        out_camera_rot = AdaptiveSFT(camera_rot,tmp_keyframes)
        out_camera_dis = camera_dis * 0

        out_camera_z = []
        out_camera_y = []
        out_camera_x = [] 
        max_frame = len(camera_dis)
        for i in tqdm(range(max_frame)):
            view = glm.mat4(1.0)
            view = glm.translate(view,glm.vec3(0,0,abs(out_camera_dis[i])))
            rot = glm.mat4(1.0)
            rot = glm.rotate(rot,out_camera_rot[i][1],glm.vec3(0,1,0))
            rot = glm.rotate(rot,out_camera_rot[i][2],glm.vec3(0,0,-1))
            # print(rot)
            rot = glm.rotate(rot,out_camera_rot[i][0],glm.vec3(1,0,0))


            view = rot * view
            # eye_i = glm.vec3(view[3]) + tmp_camera_pos[i]* glm.vec3(1, 1, -1)
            z_i = glm.normalize(glm.mat3(view) * glm.vec3(0, 0, -1))
            y_i = glm.normalize(glm.mat3(view) * glm.vec3(0, 1, 0))
            x_i = glm.normalize(glm.mat3(view) * glm.vec3(1, 0, 0))

            # out_camera_eye.append(list(eye_i))
            out_camera_z.append(list(z_i))
            out_camera_y.append(list(y_i))
            out_camera_x.append(list(x_i))

        with open(os.path.join(camera_filtered_dir,camera_file), 'w') as ocf:
            json.dump({
                'Fov':out_camera_fov.tolist(),
                'camera_eye':out_camera_eye.tolist(),
                'camera_z':out_camera_z,
                'camera_y':out_camera_y,
                'camera_x':out_camera_x,
                'camera_rot':out_camera_rot.tolist()
            },ocf)
    
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='renders/test/exp5_long/longexp5-3000-gw-1.75-1/CameraCentric')
parser.add_argument('--output_dir', type=str, default='renders/test/exp5_long/longexp5-3000-gw-1.75-1/CameraFiltered')
args = parser.parse_args()

if __name__ == "__main__":
    FilterCamera(args)