import os
import glob
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from .preprocess import Normalizer, vectorize_many

class AMCDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "jukebox",
        normalizer_pose: Any = None,
        normalizer_camera_dis: Any = None,
        normalizer_camera_pos: Any = None,
        normalizer_camera_rot: Any = None,
        normalizer_camera_fov: Any = None,
        normalizer_camera_eye: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False
    ):
        self.data_path = data_path
        self.data_fps = 30

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer_pose = normalizer_pose
        self.normalizer_camera_dis = normalizer_camera_dis
        self.normalizer_camera_pos = normalizer_camera_pos
        self.normalizer_camera_rot = normalizer_camera_rot
        self.normalizer_camera_fov = normalizer_camera_fov
        self.normalizer_camera_eye = normalizer_camera_eye
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer_pose, open(os.path.join(backup_path, "normalizer_pose.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_dis, open(os.path.join(backup_path, "normalizer_camera_dis.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_pos, open(os.path.join(backup_path, "normalizer_camera_pos.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_rot, open(os.path.join(backup_path, "normalizer_camera_rot.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_fov, open(os.path.join(backup_path, "normalizer_camera_fov.pkl"), "wb")
            )
            pickle.dump(
                normalizer_camera_eye, open(os.path.join(backup_path, "normalizer_camera_eye.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            data = self.load_amc()  # Call this last
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        # print(
        #     f"Loaded {self.name} Dataset With Dimensions: Pose: {data['pos'].shape}, camera_eye: {data['c_eye'].shape}, camera_z: {data['c_z'].shape}, camera_y: {data['c_y'].shape}, camera_x: {data['c_x'].shape},camera_pos: {data['c_pos'].shape}, camera_rot: {data['c_rot'].shape}, camera_dis: {data['c_dis'].shape}, camera_fov: {data['Fov'].shape}, bone_mask: {data['b_mask'].shape}"
        # )
        print(
            f"Loaded {self.name} Dataset With Dimensions: Pose: {data['pos'].shape}, camera_dis: {data['c_dis'].shape}, camera_pos: {data['c_pos'].shape}, camera_rot: {data['c_rot'].shape}, camera_fov: {data['c_fov'].shape}, camera_eye: {data['c_eye'].shape}, bone_mask: {data['b_mask'].shape}"
        )

        # process data
        motion_n_camera, bone_mask = self.process_dataset(data["pos"], data['c_dis'], data['c_pos'], data['c_rot'], data['c_fov'], data['c_eye'], data['b_mask'])
        
        self.data = {
            "motion": motion_n_camera[:,:,:60*3],
            "camera": motion_n_camera[:,:,60*3:],
            "bone_mask": bone_mask,
            "filenames": data['filenames'],
            "wavs": data['wavs'],
        }
        assert len(motion_n_camera) == len(data['b_mask']) == len(data['filenames'])
        self.length = len(motion_n_camera)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        return self.data["camera"][idx], self.data["bone_mask"][idx], self.data["motion"][idx], feature, filename_, self.data["wavs"][idx]

    def load_amc(self):
        # open data path
        split_data_path = os.path.join(
            self.data_path, "Train" if self.train else "Test"
        )

        # Structure:
        # data
        #   |- Train
        #   |    |- Audio
        #   |    |- Audio_sliced
        #   |    |- CameraCentric
        #   |    |- CameraCentric_sliced
        #   |    |- jukebox_feats
        #   |    |- Keypoints3D_sliced
        #   |    |- Simplified_MotionGlobalTransform
        #   |    |- Simplified_MotionGlobalTransform_sliced

        motion_path = os.path.join(split_data_path, "Keypoints3D_sliced")
        camera_path = os.path.join(split_data_path, "CameraCentric_sliced")
        bone_mask_path = os.path.join(split_data_path, "BoneMask_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, f"Audio_sliced")
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.json")))
        cameras = sorted(glob.glob(os.path.join(camera_path, "*.json")))
        bone_masks = sorted(glob.glob(os.path.join(bone_mask_path, "*.json")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the motions, cameras and features together
        all_pos = []
        all_c_dis = []
        all_c_pos = []
        all_c_rot = []
        all_c_fov = []
        all_c_eye = []
        all_mask = []
        all_names = []
        all_wavs = []
        assert len(motions) == len(cameras) == len(bone_masks) == len(features) == len(wavs)
        for motion, camera, bone_mask, feature, wav in tqdm(zip(motions, cameras, bone_masks, features, wavs)):
            # make sure name is matching
            m_name = os.path.splitext(os.path.basename(motion))[0]
            c_name = os.path.splitext(os.path.basename(camera))[0]
            bm_name = os.path.splitext(os.path.basename(bone_mask))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            # print(m_name[1:].split('kps3D')[0][:-1], c_name[1:].split('slice')[0][:-1], f_name[1:].split('slice')[0][:-3], w_name[1:].split('slice')[0][:-3], m_name[1:].split('slice')[1], c_name[1:].split('slice')[1], f_name[1:].split('slice')[1], w_name[1:].split('slice')[1])
            assert m_name[1:].split('kps3D')[0][:-1] == c_name[1:].split('slice')[0][:-1] == bm_name[2:].split('slice')[0][:-1] == f_name[1:].split('slice')[0][:-3] == w_name[1:].split('slice')[0][:-3], m_name[1:].split('slice')[1] == c_name[1:].split('slice')[1] == bm_name[2:].split('slice')[1] == f_name[1:].split('slice')[1] == w_name[1:].split('slice')[1]
            assert str((motion, camera, bone_mask, feature, wav))

            # load motion
            with open(motion, 'r') as mf:
                motion_data = json.load(mf)
            pos = motion_data["Keypoints3D"]
            # load camera
            with open(camera, 'r') as cf:
                camera_data = json.load(cf)
            c_dis = camera_data['Distance']
            c_pos = camera_data['Position']
            c_rot = camera_data['Rotation']
            c_fov = camera_data['Fov']
            c_eye = camera_data['camera_eye']
            # load bone mask
            with open(bone_mask, 'r') as bmf:
                bone_mask_data = json.load(bmf)
            b_mask = bone_mask_data['bone_mask']
            all_pos.append(pos)
            all_c_dis.append(c_dis)
            all_c_pos.append(c_pos)
            all_c_rot.append(c_rot)
            all_c_fov.append(c_fov)
            all_c_eye.append(c_eye)
            all_mask.append(b_mask)
            all_names.append(feature)
            all_wavs.append(wav)
        all_pos = np.array(all_pos)  # N x seq x (joint * 3)
        all_c_dis = np.array(all_c_dis)  # N x seq x 3
        all_c_pos = np.array(all_c_pos)  # N x seq x 3
        all_c_rot = np.array(all_c_rot)  # N x seq x 3
        all_c_fov = np.array(all_c_fov)  # N x seq x 1
        all_c_eye = np.array(all_c_eye)  # N x seq x 3
        all_mask = np.array(all_mask)  # N x seq x joint
        # downsample the motions to the data fps
        print(all_pos.shape)
        data = {"pos": all_pos, "c_dis": all_c_dis, "c_pos": all_c_pos, "c_rot": all_c_rot, "c_fov": all_c_fov, "c_eye": all_c_eye, "b_mask": all_mask, "filenames": all_names, "wavs": all_wavs}
        return data

    def process_dataset(self, kps_pos, camera_dis, camera_pos, camera_rot, camera_fov, camera_eye, b_mask):
        for dis_i in camera_dis:
            if(not len(dis_i) == 150):
                print(len(dis_i))
        for fov_i in camera_fov:
            if(not len(fov_i) == 150):
                print(len(fov_i))
        kps_pos = torch.Tensor(np.array(kps_pos,dtype='float32'))
        camera_dis = torch.Tensor(np.array(camera_dis,dtype='float32'))
        camera_pos = torch.Tensor(np.array(camera_pos,dtype='float32'))
        camera_rot = torch.Tensor(np.array(camera_rot,dtype='float32'))
        camera_fov = torch.Tensor(np.array(camera_fov,dtype='float32'))
        camera_eye = torch.Tensor(np.array(camera_eye,dtype='float32'))
        b_mask = torch.Tensor(np.array(b_mask,dtype='float32'))
        # now, flatten everything into: batch x sequence x [...]
        pose_vec_input = vectorize_many([kps_pos]).float().detach()
        camera_dis_vec_input = vectorize_many([camera_dis]).float().detach()
        camera_pos_vec_input = vectorize_many([camera_pos]).float().detach()
        camera_rot_vec_input = vectorize_many([camera_rot]).float().detach()
        camera_fov_vec_input = vectorize_many([camera_fov]).float().detach()
        camera_eye_vec_input = vectorize_many([camera_eye]).float().detach()
        b_mask_vec_input = vectorize_many([b_mask]).float().detach()

        # normalize the data. Both train and test need the same normalizer.
        if self.train:
            self.normalizer_pose = Normalizer(pose_vec_input)
            self.normalizer_camera_dis = Normalizer(camera_dis_vec_input)
            self.normalizer_camera_pos = Normalizer(camera_pos_vec_input)
            self.normalizer_camera_rot = Normalizer(camera_rot_vec_input)
            self.normalizer_camera_fov = Normalizer(camera_fov_vec_input)
            self.normalizer_camera_eye = Normalizer(camera_eye_vec_input)
        else:
            assert self.normalizer_pose is not None
            assert self.normalizer_camera_dis is not None
            assert self.normalizer_camera_pos is not None
            assert self.normalizer_camera_rot is not None
            assert self.normalizer_camera_fov is not None
            assert self.normalizer_camera_eye is not None
        pose_vec_input = self.normalizer_pose.normalize(pose_vec_input)
        camera_dis_vec_input = self.normalizer_camera_dis.normalize(camera_dis_vec_input)
        camera_pos_vec_input = self.normalizer_camera_pos.normalize(camera_pos_vec_input)
        camera_rot_vec_input = self.normalizer_camera_rot.normalize(camera_rot_vec_input)
        camera_fov_vec_input = self.normalizer_camera_fov.normalize(camera_fov_vec_input)
        camera_eye_vec_input = self.normalizer_camera_eye.normalize(camera_eye_vec_input)

        pose_camera_vec_input = torch.cat([pose_vec_input, camera_dis_vec_input, camera_pos_vec_input, camera_rot_vec_input, camera_fov_vec_input, camera_eye_vec_input], dim = 2)
        assert not torch.isnan(pose_camera_vec_input).any()
        data_name = "Train" if self.train else "Test"

        # cut the dataset
        if self.data_len > 0:
            pose_camera_vec_input = pose_camera_vec_input[: self.data_len]
            b_mask_vec_input = b_mask_vec_input[: self.data_len]

        pose_camera_vec_input = pose_camera_vec_input

        print(f"{data_name} Dataset Motion Features Dim: {pose_camera_vec_input.shape}")

        return pose_camera_vec_input,b_mask_vec_input

