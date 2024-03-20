# check if the downloaded data is the same size as raw data of DCM
import os
import argparse
import json

def GetRawIndex(e):
    return int(e[3:])

def FileSize(prefix,suffix,data_dir):
    dir_list = os.listdir(data_dir)
    dir_list.sort(key=GetRawIndex)
    size_dict = {}
    for i_dir in dir_list:
        file_name = prefix+i_dir[3:]+suffix
        file_path = os.path.join(data_dir, i_dir, file_name)
        size_dict[prefix+i_dir[3:]+suffix] = os.stat(file_path).st_size
    return size_dict

def ExportFileSize(args):
    size_audio = FileSize('a','.wav',args.dcm_raw_dir)
    size_camera = FileSize('c','.vmd',args.dcm_raw_dir)
    size_motion = FileSize('m','.vmd',args.dcm_raw_dir)
    with open(args.file_size_json, 'w') as fsf:
        json.dump({
            'audio_file_size':size_audio,
            'camera_file_size':size_camera,
            'motion_file_size':size_motion
        },fsf)

def CheckFileSize(args):
    size_audio = FileSize('a','.wav',args.dcm_raw_dir)
    size_camera = FileSize('c','.vmd',args.dcm_raw_dir)
    size_motion = FileSize('m','.vmd',args.dcm_raw_dir)
    with open(args.file_size_json, 'r') as fsf:
        read_file_size = json.load(fsf)
    audio_differ = set(size_audio.items()) ^ set(read_file_size['audio_file_size'].items())
    if len(audio_differ) == 0:
        print('Audio .wav files check!')
    else:
        print('The following audio files have difference')
        print(audio_differ)
    camera_differ = set(size_camera.items()) ^ set(read_file_size['camera_file_size'].items())
    if len(camera_differ) == 0:
        print('Camera .vmd files check!')
    else:
        print('The following camera files have difference')
        print(camera_differ)
    motion_differ = set(size_motion.items()) ^ set(read_file_size['motion_file_size'].items())
    if len(motion_differ) == 0:
        print('Motion .vmd files check!')
    else:
        print('The following motion files have difference')
        print(motion_differ)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_raw_dir', type=str, default='')
    parser.add_argument('--file_size_json', type=str, default='')
    args = parser.parse_args()

    # ExportFileSize(args)
    CheckFileSize(args)