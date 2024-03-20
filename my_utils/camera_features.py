import numpy as np
from my_utils.mmd_skeleton import Mmd_Skeleton
from tqdm import tqdm
import math
import sys

def extract_camera_shot_features_new(camera_data, pose_data, bone_mask_data):
    features = CameraShotFeatures(camera_data, pose_data, bone_mask_data)
    camera_shot_feature_vector = np.concatenate([features.body_inside_divide_shot[:-1].reshape(-1,1),features.body_inside_divide_body[:-1].reshape(-1,1)], axis=-1)
    camera_shot_feature_vector = np.array(camera_shot_feature_vector, dtype=np.float32)
    return camera_shot_feature_vector, features

def extract_camera_kinetic_features(camera_data): 
    features = CameraKineticFeatures(camera_data)
    camera_kinetic_feature_vector = []
    feature_vector = np.hstack(
        [
            features.average_velocity_energy(features.camera_eye),
            features.average_acceleration_energy_expenditure(features.camera_eye),
            features.average_velocity_energy(features.camera_z),
            features.average_acceleration_energy_expenditure(features.camera_z),
            features.average_velocity_energy(features.camera_y),
            features.average_acceleration_energy_expenditure(features.camera_y),
            features.average_velocity_energy(features.camera_x),
            features.average_acceleration_energy_expenditure(features.camera_x),
            features.average_velocity_energy(features.fov),
            features.average_acceleration_energy_expenditure(features.fov),
        ]
    )
    camera_kinetic_feature_vector.extend(feature_vector)
    camera_kinetic_feature_vector = np.array(camera_kinetic_feature_vector, dtype=np.float32)
    return camera_kinetic_feature_vector

def calc_average_velocity(subjects, i, sliding_window, frame_time):
    current_window = 0
    if len(subjects.shape) == 1:
        average_velocity = np.zeros(1)
    else:
        average_velocity = np.zeros(len(subjects[0]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j >= len(subjects):
            continue
        average_velocity += (
            subjects[i + j] - subjects[i + j - 1]
        )
        current_window += 1
    return np.linalg.norm(average_velocity / (current_window * frame_time))

def calc_average_acceleration(
    subjects, i, sliding_window, frame_time
):
    current_window = 0
    if len(subjects.shape) == 1:
        average_acceleration = np.zeros(1)
    else:
        average_acceleration = np.zeros(len(subjects[0]))
    for j in range(-sliding_window, sliding_window + 1):
        if i + j - 1 < 0 or i + j + 1 >= len(subjects):
            continue
        v2 = (
            subjects[i + j + 1] - subjects[i + j]
        ) / frame_time
        v1 = (
            subjects[i + j]
            - subjects[i + j - 1]
        ) / frame_time
        average_acceleration += (v2 - v1) / frame_time
        current_window += 1
    return np.linalg.norm(average_acceleration / current_window)


class CameraShotFeatures:
    def __init__(
        self, camera_data, pose_data, bone_mask_data, frame_time=1./30, sliding_window=2
    ):
        self.camera_eye = np.array(camera_data['camera_eye'], dtype=np.float32)#(seq_len,3)
        self.camera_z = np.array(camera_data['camera_z'], dtype=np.float32)#(seq_len,3)
        self.camera_y = np.array(camera_data['camera_y'], dtype=np.float32)#(seq_len,3)
        self.camera_x = np.array(camera_data['camera_x'], dtype=np.float32)#(seq_len,3)
        self.fov = np.array(camera_data['Fov'], dtype=np.float32)#(seq_len,1)
        self.cos_fov = np.cos(self.fov*0.5/180 * math.pi)
        self.tan_fov = np.tan(self.fov*0.5/180 * math.pi)
        self.frame_cnt = len(self.fov)
        self.pose = np.array(pose_data['Keypoints3D'], dtype=np.float32)[:self.frame_cnt]#(seq_len,kps*3)
        self.bone_mask = np.array(bone_mask_data['bone_mask'], dtype=np.float32)[:self.frame_cnt]
        self.joint_num = self.bone_mask.shape[1]
        self.body_inside_shot_projection_area = np.zeros(self.frame_cnt)
        self.body_projection_area = np.zeros(self.frame_cnt)
        self.calc_body_inside_shot_projection_area()
        self.calc_body_projection_area()
        self.body_inside_divide_shot = self.body_inside_shot_projection_area / ((4*self.tan_fov*self.tan_fov).reshape(-1) + 1e-20)
        self.body_inside_divide_body = self.body_inside_shot_projection_area / (self.body_projection_area + 1e-20)
        if np.nan in self.body_inside_divide_shot:
            print('nan in body_inside_divide_shot')
            sys.exit(1)
        if np.nan in self.body_inside_divide_body:
            print('nan in body_inside_divide_body')
            sys.exit(1)
        self.frame_time = frame_time
        self.sliding_window = sliding_window
    
    def calc_body_inside_shot_projection_area(self):# add keypoints edge constraints
        # print('calc_inside_shot_projection_area……')
        pose2eye = self.pose.reshape(self.frame_cnt, -1, 3).transpose(1,0,2) - self.camera_eye##(kps, seq_len, 3)
        pose2z = np.sum(pose2eye*self.camera_z,axis = -1)
        pose2y = np.sum(pose2eye*self.camera_y,axis = -1)
        pose2x = np.sum(pose2eye*self.camera_x,axis = -1)
        # print(pose2z.shape, pose2y.shape)
        pose2y_z = pose2y / (pose2z + 1e-20)
        pose2x_z = pose2x / (pose2z + 1e-20)

        m_skeleton = Mmd_Skeleton()

        for i in range(self.frame_cnt):
            cur_x_min = 1000000.0
            cur_x_max = -1000000.0
            cur_y_min = 1000000.0
            cur_y_max = -1000000.0
            
            bone_exist = False
            
            for j in range(self.joint_num):
                if self.bone_mask[i][j] == 0:#this joint is not inside camera field of view
                    continue
                else:
                    bone_exist = True
                    # print(pose2y_z[j][i], pose2x_z[j][i])
                    if pose2y_z[j][i] > cur_y_max:
                        cur_y_max = pose2y_z[j][i]
                    if pose2y_z[j][i] < cur_y_min:
                        cur_y_min = pose2y_z[j][i]
                    if pose2x_z[j][i] > cur_x_max:
                        cur_x_max = pose2x_z[j][i]
                    if pose2x_z[j][i] < cur_x_min:
                        cur_x_min = pose2x_z[j][i]
                    # print(cur_y_max, cur_y_min, cur_x_max, cur_x_min)
            if bone_exist == True:
                self.body_inside_shot_projection_area[i] = (cur_y_max - cur_y_min)*(cur_x_max - cur_x_min)
            else:
                self.body_inside_shot_projection_area[i] = 0.0


    def calc_body_projection_area(self):
        # print('calc_body_projection_area……')
        pose2eye = self.pose.reshape(self.frame_cnt, -1, 3).transpose(1,0,2) - self.camera_eye##(kps, seq_len, 3)
        pose2z = np.sum(pose2eye*self.camera_z,axis = -1)
        pose2y = np.sum(pose2eye*self.camera_y,axis = -1)
        pose2x = np.sum(pose2eye*self.camera_x,axis = -1)
        
        pose2y_z = pose2y / (pose2z + 1e-20)
        pose2x_z = pose2x / (pose2z + 1e-20)

        for i in range(self.frame_cnt):
            cur_x_min = 0.0
            cur_x_max = 0.0
            cur_y_min = 0.0
            cur_y_max = 0.0
            j = 0
            while(pose2z[j][i] <= 0):
                j += 1
                if j == self.joint_num:
                    j -= 1
                    break 
            cur_x_min = pose2x_z[j][i]
            cur_x_max = pose2x_z[j][i]
            cur_y_min = pose2y_z[j][i]
            cur_y_max = pose2y_z[j][i]
            j += 1
            while(j < self.joint_num):
                if pose2z[j][i] <= 0:
                    j += 1
                    continue
                if pose2y_z[j][i] > cur_y_max:
                    cur_y_max = pose2y_z[j][i]
                if pose2y_z[j][i] < cur_y_min:
                    cur_y_min = pose2y_z[j][i]
                if pose2x_z[j][i] > cur_x_max:
                    cur_x_max = pose2x_z[j][i]
                if pose2x_z[j][i] < cur_x_min:
                    cur_x_min = pose2x_z[j][i]
                j += 1
            self.body_projection_area[i] = (cur_y_max - cur_y_min)*(cur_x_max - cur_x_min)

class CameraKineticFeatures:
    def __init__(
        self, camera_data, frame_time=1./30, sliding_window=2
    ):
        self.camera_eye = np.array(camera_data['camera_eye'], dtype=np.float32)
        self.camera_z = np.array(camera_data['camera_z'], dtype=np.float32)
        self.camera_y = np.array(camera_data['camera_y'], dtype=np.float32)
        self.camera_x = np.array(camera_data['camera_x'], dtype=np.float32)
        self.fov = np.array(camera_data['Fov'], dtype=np.float32)
        self.frame_time = frame_time
        self.sliding_window = sliding_window

    def average_velocity_energy(self, subject):
        average_v_kinetic_energy = 0
        for i in range(1, len(subject)):
            average_velocity = calc_average_velocity(
                subject, i, self.sliding_window, self.frame_time
            )
            average_v_kinetic_energy += average_velocity ** 2
        average_v_kinetic_energy = average_v_kinetic_energy / (
            len(subject) - 1.0
        )
        return average_v_kinetic_energy

    def average_acceleration_energy_expenditure(self, subject):
        average_acceleration = 0.0
        for i in range(1, len(subject)):
            average_acceleration += calc_average_acceleration(
                subject, i, self.sliding_window, self.frame_time
            )
        average_acceleration = average_acceleration / (len(subject) - 1.0)
        return average_acceleration
