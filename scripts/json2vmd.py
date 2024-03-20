import sys
sys.path.append("..")
sys.path.append(".")
from my_utils.vmd import Vmd
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, default='output/amc_motion_correct_json')
parser.add_argument('--vmd_dir', type=str, default='output/amc_motion_correct_vmd')
parser.add_argument('--data_type', type=str, default='motion')


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.vmd_dir):
        os.makedirs(args.vmd_dir, exist_ok=True)
    motion_jsons = os.listdir(args.json_dir)
    for m_json in motion_jsons:
        m_path = os.path.join(args.json_dir,m_json)
        out_path = os.path.join(args.vmd_dir,m_json[:-5]+'.vmd')
        if args.data_type == 'motion':
            Vmd.saba_motion_json_to_vmd(m_path,out_path)
        elif args.data_type == 'camera':
            Vmd.saba_camera_json_to_vmd(m_path,out_path) 