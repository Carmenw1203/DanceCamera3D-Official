# After we split the long data into smaller pieces we could finally split data by catgories

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


def SplitByCategories(args):
    with open(args.music_categories, 'r') as f:
        music_categories = json.load(f)
    short_music_categories = music_categories.copy()
    for item in short_music_categories:
        short_music_categories[item] = []
    audio_list = os.listdir(args.audio_dir)
    for audio_i in tqdm(audio_list):
        audio_index = int(audio_i[1:-4].split('_')[0])
        for category in music_categories:
            if music_categories[category].count(audio_index):
                short_music_categories[category].append(audio_i[1:-4])
    Path(f'{args.output_dir}/Train/Audio').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Train/CameraCentric').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Train/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Validation/Audio').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Validation/CameraCentric').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Validation/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Test/Audio').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Test/CameraCentric').mkdir(parents=True, exist_ok=False)
    Path(f'{args.output_dir}/Test/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=False)
    train_list = []
    validation_list = []
    test_list = []
    for category in short_music_categories:
        len_category = len(short_music_categories[category])
        sign_category = category[:1]
        
        rd = np.random.choice(['Train','Validation', 'Test'],len_category,p=[0.8,0.1,0.1])
        for i in range(len_category):
            cur_index = short_music_categories[category][i]
            audio_path = f'{args.audio_dir}/a{cur_index}.wav'
            camera_c_path = f'{args.camera_c_dir}/c{cur_index}.json'
            motion_path = f'{args.motion_dir}/m{cur_index}_gt.json'
            if rd[i] == 'Train':
                train_list.append(sign_category+"_"+cur_index)
                shutil.copyfile(audio_path, f'{args.output_dir}/Train/Audio/a{cur_index}_{sign_category}.wav')
                shutil.copyfile(camera_c_path, f'{args.output_dir}/Train/CameraCentric/c{cur_index}.json')
                shutil.copyfile(motion_path, f'{args.output_dir}/Train/Simplified_MotionGlobalTransform/m{cur_index}_gt.json')
            elif rd[i] == 'Validation':
                validation_list.append(sign_category+"_"+cur_index)
                shutil.copyfile(audio_path, f'{args.output_dir}/Validation/Audio/a{cur_index}_{sign_category}.wav')
                shutil.copyfile(camera_c_path, f'{args.output_dir}/Validation/CameraCentric/c{cur_index}.json')
                shutil.copyfile(motion_path, f'{args.output_dir}/Validation/Simplified_MotionGlobalTransform/m{cur_index}_gt.json')
            elif rd[i] == 'Test':
                test_list.append(sign_category+"_"+cur_index)
                shutil.copyfile(audio_path, f'{args.output_dir}/Test/Audio/a{cur_index}_{sign_category}.wav')
                shutil.copyfile(camera_c_path, f'{args.output_dir}/Test/CameraCentric/c{cur_index}.json')
                shutil.copyfile(motion_path, f'{args.output_dir}/Test/Simplified_MotionGlobalTransform/m{cur_index}_gt.json')
    with open(args.split_train, 'w') as f:
        json.dump(train_list,f,indent=2)  
    with open(args.split_validation, 'w') as f:
        json.dump(validation_list,f,indent=2)  
    with open(args.split_test, 'w') as f:
        json.dump(test_list,f,indent=2)  
    # print(a)
def CopyFiles(set_tag, id_list, args):
   
    for index_i in id_list:
        sign_category = index_i[0]
        cur_index = index_i[2:]
        audio_path = f'{args.audio_dir}/a{cur_index}.wav'
        camera_c_path = f'{args.camera_c_dir}/c{cur_index}.json'
        motion_path = f'{args.motion_dir}/m{cur_index}_gt.json'
        shutil.copyfile(audio_path, f'{args.output_dir}/{set_tag}/Audio/a{cur_index}_{sign_category}.wav')
        shutil.copyfile(camera_c_path, f'{args.output_dir}/{set_tag}/CameraCentric/c{cur_index}.json')
        shutil.copyfile(motion_path, f'{args.output_dir}/{set_tag}/Simplified_MotionGlobalTransform/m{cur_index}_gt.json')
def SplitByCategoriesFromFile(args):
    Path(f'{args.output_dir}/Train/Audio').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Train/CameraCentric').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Train/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Validation/Audio').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Validation/CameraCentric').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Validation/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Test/Audio').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Test/CameraCentric').mkdir(parents=True, exist_ok=True)
    Path(f'{args.output_dir}/Test/Simplified_MotionGlobalTransform').mkdir(parents=True, exist_ok=True)

    with open(args.split_train, 'r') as trainf:
        train_list = json.load(trainf)
    with open(args.split_validation, 'r') as valf:
        validation_list = json.load(valf)
    with open(args.split_test, 'r') as testf:
        test_list = json.load(testf)
    CopyFiles('Train', train_list, args)
    CopyFiles('Validation', validation_list, args)
    CopyFiles('Test', test_list, args)
parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, default='DCM_data/amc_aligned_data_split/Audio')
parser.add_argument('--camera_c_dir', type=str, default='DCM_data/amc_aligned_data_split/CameraCentric')
parser.add_argument('--motion_dir', type=str, default='DCM_data/amc_aligned_data_split/Simplified_MotionGlobalTransform/')
parser.add_argument('--music_categories', type=str, default='DCM_data/split/music_categories.json')
parser.add_argument('--split_train', type=str, default='DCM_data/split/train.json')
parser.add_argument('--split_validation', type=str, default='DCM_data/split/validation.json')
parser.add_argument('--split_test', type=str, default='DCM_data/split/test.json')
parser.add_argument('--output_dir', type=str, default='DCM_data/amc_data_split_by_categories')
parser.add_argument('--split_from_file', type=bool, default=True)
args = parser.parse_args()


if __name__ == '__main__':
    if args.split_from_file:
        SplitByCategoriesFromFile(args)
    else:
        SplitByCategories(args)
    