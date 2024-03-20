import os
import sys
sys.path.append("..")
sys.path.append(".")
import json
import glob
import shutil
from scipy import linalg
from my_utils.detect_bone_mask import DetactBoneMask
from my_utils.slice import GlobalTransform2Keypoints_File
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from my_utils.camera_features import extract_camera_kinetic_features, extract_camera_shot_features_new
from audio_extraction.baseline_features import extract_folder as baseline_extract
from matplotlib import pyplot as plt

def GetBoneMaskNameIndex(e):
    
    e_name = e.split(os.sep)[-1].split('.')[0]
    e_name_split = e_name[2:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def GetNameIndex(e):
    
    e_name = e.split(os.sep)[-1].split('.')[0]
    e_name_split = e_name[1:].split('_')
    m_index = e_name_split[0]
    if len(e_name_split) >= 2:
        s_index = e_name_split[1]
    else:
        s_index = '0'
    return m_index+'_'+s_index

def DancerMissingRate(bone_mask_list):
    frame_num = 0.0
    dancer_missing_num = 0.0
    dancer_missing_rate = 0.0

    for bone_mask_file in bone_mask_list:
        with open(bone_mask_file, 'r') as bmf:
            bone_mask_data = json.load(bmf)
        np_bone_mask = np.sum(np.array(bone_mask_data["bone_mask"]),axis = -1)
        frame_num += np_bone_mask.shape[0]
        dancer_missing_num += np.sum(np_bone_mask == 0)
    dancer_missing_rate = dancer_missing_num / frame_num
    return dancer_missing_rate

def calc_and_save_kinetic_feats(camera_c_list, camera_kfeature_dir):
    for camera_c_file in tqdm(camera_c_list):
        file_base = os.path.basename(camera_c_file)
        with open(camera_c_file, 'r') as ccf:
            camera_c_data = json.load(ccf)
        
        tmp_frame_cnt = int(len(camera_c_data['camera_eye']))
        tmp_cnt = 0
        while(tmp_cnt*75 + 75 <= tmp_frame_cnt):#75 denotes 2.5s with 30 FPS
            tmp_camera_c_data = {}
            tmp_camera_c_data['camera_eye'] = camera_c_data['camera_eye'][tmp_cnt*75:tmp_cnt*75 + 75]
            tmp_camera_c_data['camera_z'] = camera_c_data['camera_z'][tmp_cnt*75:tmp_cnt*75 + 75]
            tmp_camera_c_data['camera_y'] = camera_c_data['camera_y'][tmp_cnt*75:tmp_cnt*75 + 75]
            tmp_camera_c_data['camera_x'] = camera_c_data['camera_x'][tmp_cnt*75:tmp_cnt*75 + 75]
            tmp_camera_c_data['Fov'] = camera_c_data['Fov'][tmp_cnt*75:tmp_cnt*75 + 75]
            kfeature_save_name = 'k'+file_base[1:-5]+'_'+str(tmp_cnt)+'.npy'
            np.save(os.path.join(camera_kfeature_dir, kfeature_save_name), extract_camera_kinetic_features(tmp_camera_c_data))
            tmp_cnt += 1

def calc_and_save_shot_feats(camera_c_list, motion_list, bone_mask_list, camera_sfeature_dir, camera_sfeatures_plot_dir):
    assert len(camera_c_list) == len(motion_list)
    f_cnt = len(camera_c_list)
    for f_i in tqdm(range(f_cnt)):
        with open(camera_c_list[f_i], 'r') as ccf:
            camera_c_data = json.load(ccf)
        with open(motion_list[f_i], 'r') as mf:
            motion_data = json.load(mf)
        with open(bone_mask_list[f_i], 'r') as bmf:
            bone_mask_data = json.load(bmf)
        file_base = os.path.basename(camera_c_list[f_i])
        sfeature_save_name = 's'+file_base[1:-4]+'npy'
        shot_features_vector, shot_features = extract_camera_shot_features_new(camera_c_data, motion_data, bone_mask_data)
        np.save(os.path.join(camera_sfeature_dir, sfeature_save_name), shot_features_vector)

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def calc_fid(kps_gen, kps_gt):
    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt
    diff = mu1 - mu2
    eps = 1e-5
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in tqdm(range(n)):
        dist += np.sum(np.linalg.norm(feature_list[:n-i] - feature_list[i:], axis = -1))
    dist /= (n * n - n) / 2
    return dist

def LimbsCaptureDifference(bm_list1, bm_list0):
    bm_cnt = len(bm_list1)
    bm_frames_cnt = 0.0
    lcd = 0.0

    for i_bm in range(bm_cnt):
        # print(bm_list1[i_bm], bm_list0[i_bm])
        with open(bm_list1[i_bm], 'r') as f1:
            bm1_data = np.array(json.load(f1)["bone_mask"])
        with open(bm_list0[i_bm], 'r') as f0:
            bm0_data = np.array(json.load(f0)["bone_mask"])
        bm_frames_cnt += len(bm1_data)
        lcd += np.sum(np.mean((bm1_data-bm0_data[:len(bm1_data)])**2, axis = 1))
    lcd /= bm_frames_cnt

    return lcd

def CopyFiles(file_list, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for f_i in file_list:
        shutil.copy(f_i, os.path.join(target_dir, f_i.split(os.sep)[-1]))

def calc_feature_k(root_dir):
    camera_centric_dir = os.path.join(root_dir,'CameraCentric')
    camera_kinetic_features_dir = os.path.join(root_dir,'CameraKineticFeatures')
    Path(camera_kinetic_features_dir).mkdir(parents=True, exist_ok=True)
    camera_centric_list = sorted(glob.glob(os.path.join(camera_centric_dir, "*.json")), key=GetNameIndex)
    camera_kinetic_features_list = sorted(glob.glob(os.path.join(camera_kinetic_features_dir, "*.npy")), key=GetNameIndex)
    if len(camera_kinetic_features_list) == 0:
        calc_and_save_kinetic_feats(camera_centric_list, camera_kinetic_features_dir)
        camera_kinetic_features_list = sorted(glob.glob(os.path.join(camera_kinetic_features_dir, "*.npy")), key=GetNameIndex)
    features_k = [np.load(pkl) for pkl in camera_kinetic_features_list]
    features_k = np.stack(features_k)
    
    return features_k

def calc_feature_s(root_dir):
    camera_centric_dir = os.path.join(root_dir, 'CameraCentric')
    motion_kps_dir = os.path.join(root_dir, 'Keypoints3D')
    bone_mask_dir = os.path.join(root_dir, 'BoneMask')
    camera_shot_features_dir = os.path.join(root_dir, 'CameraShotFeatures')
    camera_shot_features_plot_dir = os.path.join(root_dir, 'CameraShotFeaturesPlot')
    camera_centric_list = sorted(glob.glob(os.path.join(camera_centric_dir, "*.json")), key=GetNameIndex)
    motion_kps_list = sorted(glob.glob(os.path.join(motion_kps_dir, "*.json")), key=GetNameIndex)
    bone_mask_list = sorted(glob.glob(os.path.join(bone_mask_dir, "*.json")), key=GetNameIndex)
    
    Path(camera_shot_features_dir).mkdir(parents=True, exist_ok=True)
    Path(camera_shot_features_plot_dir).mkdir(parents=True, exist_ok=True)
    camera_shot_features_list = sorted(glob.glob(os.path.join(camera_shot_features_dir, "*.npy")), key=GetNameIndex)
    if len(camera_shot_features_list) == 0:
        calc_and_save_shot_feats(camera_centric_list, motion_kps_list, bone_mask_list, camera_shot_features_dir, camera_shot_features_plot_dir)
        camera_shot_features_list = sorted(glob.glob(os.path.join(camera_shot_features_dir, "*.npy")), key=GetNameIndex)
    features_s = [np.load(pkl) for pkl in camera_shot_features_list]
    
    features_s = np.concatenate(features_s)
    
    return features_s

def EvaluateResults(args):
    res_dir = args.result_dir
    print('Evaluation results of ' + res_dir)
    test_bone_mask_dir = os.path.join(args.test_dir,'BoneMask')
    test_bone_mask_list = sorted(glob.glob(os.path.join(test_bone_mask_dir, "*.json")), key=GetBoneMaskNameIndex)
    
    res_bone_mask_list = sorted(glob.glob(os.path.join(res_dir, "bm*.json")), key=GetBoneMaskNameIndex)
    res_camera_centric_list = sorted(glob.glob(os.path.join(res_dir, "c*.json")), key=GetNameIndex)
    res_bone_mask_dir = os.path.join(res_dir, "BoneMask")
    res_camera_centric_dir = os.path.join(res_dir, "CameraCentric")
    CopyFiles(res_bone_mask_list, res_bone_mask_dir)
    CopyFiles(res_camera_centric_list, res_camera_centric_dir)

    if len(res_bone_mask_list) == 0:
        res_bone_mask_list = sorted(glob.glob(os.path.join(res_bone_mask_dir, "bm*.json")), key=GetBoneMaskNameIndex)
        res_camera_centric_list = sorted(glob.glob(os.path.join(res_camera_centric_dir, "c*.json")), key=GetNameIndex)
    # print(len(test_bone_mask_list), len(res_bone_mask_list), len(res_camera_centric_list))
    assert len(test_bone_mask_list) == len(res_bone_mask_list) == len(res_camera_centric_list)

    # fid_k kinetic
    res_features_k = calc_feature_k(args.result_dir)
    test_features_k = calc_feature_k(args.test_dir)

    test_features_kn, res_features_kn = normalize(test_features_k, res_features_k)
    fid_k_res = calc_fid(res_features_kn, test_features_kn)#calculate distance between data distributions of results and test set instead of  data distributions of results and the whole dataset. 

    # div_k kinetic
    div_k_test = calculate_avg_distance(test_features_kn)
    div_k_res = calculate_avg_distance(res_features_kn)

    print('fid_k_res:', fid_k_res, 'div_k_res:', div_k_res, 'div_k_test:', div_k_test)
    
    # fid_s shot
    test_motion_dir = os.path.join(args.test_dir, 'Simplified_MotionGlobalTransform')
    test_motion_kps_dir = os.path.join(args.test_dir, 'Keypoints3D')
    Path(test_motion_kps_dir).mkdir(parents=True, exist_ok=True)
    test_motion_kps_list = sorted(glob.glob(os.path.join(test_motion_kps_dir, "*.json")), key=GetNameIndex)
    if len(test_motion_kps_list) == 0:
        GlobalTransform2Keypoints_File(test_motion_dir, test_motion_kps_dir)
        test_motion_kps_list = sorted(glob.glob(os.path.join(test_motion_kps_dir, "*.json")), key=GetNameIndex)
    res_motion_kps_dir = os.path.join(res_dir, "Keypoints3D")
    Path(res_motion_kps_dir).mkdir(parents=True, exist_ok=True)
    res_motion_kps_list = sorted(glob.glob(os.path.join(res_motion_kps_dir, "*.json")), key=GetNameIndex)
    if len(res_motion_kps_list) == 0:
        CopyFiles(test_motion_kps_list, res_motion_kps_dir)

    # fid_s shot
    res_features_s = calc_feature_s(args.result_dir)
    test_features_s = calc_feature_s(args.test_dir)
    
    test_features_sn, res_features_sn = normalize(test_features_s, res_features_s)
   
    fid_s_res = calc_fid(res_features_sn, test_features_sn)
    
    
    res_metrics_path = os.path.join(args.result_dir, 'metrics_new.json')
    if os.path.exists(res_metrics_path):
        with open(res_metrics_path, 'r') as rmf:
            res_metrics = json.load(rmf)
        div_s_res = res_metrics['div_s_res']
    else:
        # div_k shot
        div_s_res = calculate_avg_distance(res_features_sn)

    test_metrics_path = os.path.join(args.test_dir, 'metrics_new.json')
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path, 'r') as tmf:
            test_metrics = json.load(tmf)
        div_s_test = test_metrics['div_s_test']
    else:
        div_s_test = calculate_avg_distance(test_features_sn)
    print('fid_s_res:', fid_s_res, 'div_s_res:', div_s_res, 'div_s_test:', div_s_test)

    # dancer_missing_rate
    test_dmr = DancerMissingRate(test_bone_mask_list)
    res_dmr = DancerMissingRate(res_bone_mask_list)
    print('dancer_missing_rate: ', res_dmr, 'test_set:', test_dmr)

    # limbs capture difference
    lc_diff = LimbsCaptureDifference(res_bone_mask_list, test_bone_mask_list)
    print('limbs capture difference: ', lc_diff)


    with open(res_metrics_path, 'w') as rmwf:
        json.dump({'fid_k_res':fid_k_res, 'div_k_res':div_k_res,'fid_s_res':fid_s_res, 'div_s_res':div_s_res, 'res_dmr': res_dmr,'lcd': lc_diff},rmwf)
    with open(test_metrics_path, 'w') as tmwf:
        json.dump({'div_k_test':div_k_test, 'div_s_test':div_s_test},tmwf)
    


    
    
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--test_dir', type=str, default='DCM_data/amc_data_split_by_categories/Test')
args = parser.parse_args()

if __name__ == "__main__":
    EvaluateResults(args)