import os
import json
import sys
sys.path.append("..")
sys.path.append(".")
from model.VmdBezier import VmdBezier

import argparse
from tqdm import tqdm

import glm
import math

def FrameTimeOrder(e):
    return e["FrameTime"]


def camera_interpolation_vmd(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    input_jsons = os.listdir(input_dir)
    for i_json in input_jsons:
        i_path = os.path.join(input_dir, i_json)
        o_path = os.path.join(output_dir, i_json)
        with open(i_path, 'r') as fin:
            camera_data = json.load(fin)
        keyframe_number = camera_data["CameraKeyFrameNumber"]
        key_frames = camera_data["CameraKeyFrameRecord"]
        key_frames.sort(key=FrameTimeOrder)

        

        max_frame = key_frames[-1]["FrameTime"]
        for kf in key_frames:
            kf["Beizer"] = VmdBezier()
            kf["Beizer"].SetBezier(float(kf["Curve"][0]),float(kf["Curve"][1]),float(kf["Curve"][2]),float(kf["Curve"][3]))

        frames_distance = []
        frames_position = []
        frames_rotation = []
        frames_fov = []
        k0 = 0
        k1 = 1
        f_i = 0
        for f_i in tqdm(range(max_frame)):
            if f_i <= key_frames[k0]["FrameTime"]:
                frames_distance.append(key_frames[k0]["Distance"])
                frames_position.append((key_frames[k0]["Position"]['x'],key_frames[k0]["Position"]['y'],key_frames[k0]["Position"]['z']))
                frames_rotation.append((key_frames[k0]["Rotation"]['x'],key_frames[k0]["Rotation"]['y'],key_frames[k0]["Rotation"]['z']))
                frames_fov.append(float(key_frames[k0]["ViewAngle"]))
            elif f_i < key_frames[k1]["FrameTime"]:
                t_i = (float(f_i) - float(key_frames[k0]["FrameTime"]))/(float(key_frames[k1]["FrameTime"]) - float(key_frames[k0]["FrameTime"]))
                i_y = key_frames[k0]["Beizer"].EvalYfromTime(t_i)
                distance_i = glm.mix(key_frames[k0]["Distance"],key_frames[k1]["Distance"],i_y)
                position_x_i = glm.mix(key_frames[k0]["Position"]['x'],key_frames[k1]["Position"]['x'],i_y)
                position_y_i = glm.mix(key_frames[k0]["Position"]['y'],key_frames[k1]["Position"]['y'],i_y)
                position_z_i = glm.mix(key_frames[k0]["Position"]['z'],key_frames[k1]["Position"]['z'],i_y)
                rotation_x_i = glm.mix(key_frames[k0]["Rotation"]['x'],key_frames[k1]["Rotation"]['x'],i_y)
                rotation_y_i = glm.mix(key_frames[k0]["Rotation"]['y'],key_frames[k1]["Rotation"]['y'],i_y)
                rotation_z_i = glm.mix(key_frames[k0]["Rotation"]['z'],key_frames[k1]["Rotation"]['z'],i_y)
                fov_i = glm.mix(float(key_frames[k0]["ViewAngle"]),float(key_frames[k1]["ViewAngle"]),i_y)
                frames_distance.append(distance_i)
                frames_position.append((position_x_i,position_y_i,position_z_i))
                frames_rotation.append((rotation_x_i,rotation_y_i,rotation_z_i))
                frames_fov.append(fov_i)
            elif f_i == key_frames[k1]["FrameTime"]:
                frames_distance.append(key_frames[k1]["Distance"])
                frames_position.append((key_frames[k1]["Position"]['x'],key_frames[k1]["Position"]['y'],key_frames[k1]["Position"]['z']))
                frames_rotation.append((key_frames[k1]["Rotation"]['x'],key_frames[k1]["Rotation"]['y'],key_frames[k1]["Rotation"]['z']))
                frames_fov.append(float(key_frames[k1]["ViewAngle"]))
                k0 += 1
                k1 += 1
            # f_i += 1

        with open(o_path,'w') as fout:
            json.dump({
                "FrameTime":max_frame,
                "Distance":frames_distance,
                "Position":frames_position,
                "Rotation":frames_rotation,
                "Fov":frames_fov
            },fout)


def camera_vmd2centric(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_jsons = os.listdir(input_dir)
    for i_json in input_jsons:
        i_path = os.path.join(input_dir, i_json)
        o_path = os.path.join(output_dir, i_json)
        with open(i_path, 'r') as fin:
            camera_data = json.load(fin)
        max_frame = camera_data["FrameTime"]
        distances = camera_data["Distance"]
        positions = camera_data["Position"]
        rotations = camera_data["Rotation"]
        fov = camera_data["Fov"]

        camera_eye = []#camera_position
        camera_z = []#camera_facing_direction
        camera_y = []#camera_up_direction
        camera_x = []#camera_horizontal_direction
        camera_fov = []#camera field of view
        for i in tqdm(range(max_frame)):
            view = glm.mat4(1.0)
            view = glm.translate(view,glm.vec3(0,0,abs(distances[i])))
            rot = glm.mat4(1.0)
            rot = glm.rotate(rot,rotations[i][1],glm.vec3(0,1,0))
            rot = glm.rotate(rot,rotations[i][2],glm.vec3(0,0,-1))
            rot = glm.rotate(rot,rotations[i][0],glm.vec3(1,0,0))

            view = rot * view
            # print(type(positions[i]))
            eye_i = glm.vec3(view[3]) + positions[i]* glm.vec3(1, 1, -1)
            z_i = glm.normalize(glm.mat3(view) * glm.vec3(0, 0, -1))
            y_i = glm.normalize(glm.mat3(view) * glm.vec3(0, 1, 0))
            x_i = glm.normalize(glm.mat3(view) * glm.vec3(1, 0, 0))

            camera_eye.append(list(eye_i))
            camera_z.append(list(z_i))
            camera_y.append(list(y_i))
            camera_x.append(list(x_i))
            camera_fov.append(fov[i])
        with open(o_path,'w') as fout:
            json.dump({
                "FrameTime": max_frame,
                "camera_eye": camera_eye,
                "camera_z": camera_z,
                "camera_y": camera_y,
                "camera_x": camera_x,
                "Distance":distances,
                "Position":positions,
                "Rotation":rotations,
                "Fov":fov
            },fout)

parser = argparse.ArgumentParser()
parser.add_argument('--camera_keyframe_json_dir', type=str, default='DCM_data/amc_aligned_data/CameraKeyframe')
parser.add_argument('--camera_interpolation_json_dir', type=str, default='DCM_data/amc_aligned_data/CameraInterpolated')
parser.add_argument('--camera_centric_json_dir', type=str, default='DCM_data/amc_aligned_data/CameraCentric')

if __name__ == '__main__':
    args = parser.parse_args()
    camera_keyframe_dir = args.camera_keyframe_json_dir
    interpolate_dir = args.camera_interpolation_json_dir
    centric_dir = args.camera_centric_json_dir
    # frist conduct interpolation for raw camera data(from keyframes to each frames)
    camera_interpolation_vmd(camera_keyframe_dir, interpolate_dir)
    # then convert raw camera data--vmd represention to usable camera parameters--camera centric represention(from (distance, position, rotation, veiw angle/fov) to (camera_position--eye, camera_facing_direction--camera_z, camera_up_direction--camera_y, camera_horizontal_direction--camera_x, fov))
    camera_vmd2centric(interpolate_dir, centric_dir)