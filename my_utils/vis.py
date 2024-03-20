import os
import sys
sys.path.append("..")
import json
import argparse
from pathlib import Path
from .mmd_skeleton import Mmd_Skeleton
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
import math
from ffmpy import FFmpeg
from moviepy.editor import VideoFileClip, AudioFileClip


def video_add_audio(v_path, a_path, va_path):
    # _ext_video = os.path.basename(v_path).strip().split('.')[-1]
    # _ext_audio = os.path.basename(a_path).strip().split('.')[-1]
    
    # if _ext_audio not in ['mp3', 'wav']:
    #     raise Exception('audio format not support')
    # _codec = 'copy'
    # if _ext_audio == 'wav':
    #     _codec = 'aac'
    
    # v_path = os.path.normpath(v_path)
    # v_pathparts = v_path.split(os.sep)
    # v_pathparts[-1] = 'a'+ v_pathparts[-1]
    # v_path_res = os.path.join(*v_pathparts)

    # ff = FFmpeg(
    #     inputs={v_path: None, a_path: None},
    #     outputs={v_path_res: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    # # print(ff.cmd)
    # ff.run()
    # return result
    video = VideoFileClip(v_path)
    audio = AudioFileClip(a_path)
    video_merge = video.set_audio(audio)
    video_merge.write_videofile(va_path,audio_codec='aac')

def MotionCameraRender(pose, camera_eye, camera_z, camera_y, camera_x, camera_fov, bone_mask, epoch, out_dir, audio_path, sound):
    
    epoch_out_dir = os.path.join(out_dir, epoch)
    Path(epoch_out_dir).mkdir(parents=True, exist_ok=True)
    video_name = audio_path.split(os.sep)[-1].replace("wav","mp4")
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(epoch_out_dir, video_name)
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (800,600))

    np_keypoints = pose
    
    t, d = np_keypoints.shape
    np_keypoints = np_keypoints.reshape(t,-1,3)
    m_skeleton = Mmd_Skeleton()
    for i in tqdm(range(t)):
        xp = -np_keypoints[i].T[0].T
        yp = np_keypoints[i].T[1].T
        zp = np_keypoints[i].T[2].T
        plt.figure(figsize=(10, 7.5), dpi=80)
        # plt.figure(figsize=(100, 75), dpi=80)
        ax = plt.axes(projection='3d')
        
        radius = 40*2
        # radius = 30*2
        # radius = 20*2
        # ax.set_xlim3d([-radius, radius])
        # ax.set_ylim3d([-radius / 4, radius / 4])
        # ax.set_zlim3d([-radius / 4, radius / 2])
        # ax.set_xlim3d([-radius/6, radius/6])
        

        
        # ax.view_init(elev=15., azim=120)
        # ax.set_xlim3d([-radius/6,radius/6])
        # ax.set_ylim3d([-radius/6, radius/6])
        # ax.set_zlim3d([0,radius/3])
        # ax.view_init(elev=15., azim=90)#motion view
        # ax.view_init(elev=10., azim=90)#motion view

        # ax.set_xlim3d([-radius/2, radius/6])
        ax.set_xlim3d([-radius/6, radius/2])# for 10_2
        ax.set_ylim3d([-radius / 3, radius / 3])
        ax.set_zlim3d([-radius*2 / 9, radius *4/ 9])
        # ax.view_init(elev=60., azim=90)
        ax.view_init(elev=15., azim=120)
        ax.dist = 7.5
        # ax.dist = 3
        mask_xp = xp[np.where(bone_mask[i] == 1)]
        mask_yp = yp[np.where(bone_mask[i] == 1)]
        mask_zp = zp[np.where(bone_mask[i] == 1)]
        ax.scatter3D(mask_xp, mask_zp, mask_yp, s=5, c='black', alpha=0.5)

        for bs_i in range(len(m_skeleton.simplify_bones)):
            if m_skeleton.simplify_bone_parents[bs_i] < 0:
                continue
            ax.plot(np.hstack((xp[m_skeleton.simplify_bone_parents[bs_i]], xp[bs_i])),
                    np.hstack((zp[m_skeleton.simplify_bone_parents[bs_i]], zp[bs_i])),
                    np.hstack((yp[m_skeleton.simplify_bone_parents[bs_i]], yp[bs_i])),
                    ls='-', color=m_skeleton.simplify_bone_color[bs_i])
        c_eye = camera_eye[i]
        c_x = camera_x[i]
        c_y = camera_y[i]
        c_z = camera_z[i]

        c_eye[0] *= -1
        c_x[0] *= -1
        c_y[0] *= -1
        c_z[0] *= -1


        c_fov = camera_fov[i]
        c_fov_t = math.tan(c_fov*0.5/360 * 2 * math.pi)
        for j in range(3):
            c_x[j] *= c_fov_t
            c_y[j] *= c_fov_t
        ax.plot(np.hstack((c_eye[0], c_eye[0]+c_z[0]*200.0)),
                    np.hstack((c_eye[2], c_eye[2]+c_z[2]*200.0)),
                    np.hstack((c_eye[1], c_eye[1]+c_z[1]*200.0)),
                    ls='-', color='red')
        ax.plot(np.hstack((c_eye[0], c_eye[0]+(c_z[0]+c_y[0]+c_x[0])*200)),
                np.hstack((c_eye[2], c_eye[2]+(c_z[2]+c_y[2]+c_x[2])*200)),
                np.hstack((c_eye[1], c_eye[1]+(c_z[1]+c_y[1]+c_x[1])*200)),
                ls='-', color='k')
        ax.plot(np.hstack((c_eye[0], c_eye[0]+(c_z[0]+c_y[0]-c_x[0])*200)),
                    np.hstack((c_eye[2], c_eye[2]+(c_z[2]+c_y[2]-c_x[2])*200)),
                    np.hstack((c_eye[1], c_eye[1]+(c_z[1]+c_y[1]-c_x[1])*200)),
                    ls='-', color='k')
        ax.plot(np.hstack((c_eye[0], c_eye[0]+(c_z[0]-c_y[0]+c_x[0])*200)),
                    np.hstack((c_eye[2], c_eye[2]+(c_z[2]-c_y[2]+c_x[2])*200)),
                    np.hstack((c_eye[1], c_eye[1]+(c_z[1]-c_y[1]+c_x[1])*200)),
                    ls='-', color='k')
        ax.plot(np.hstack((c_eye[0], c_eye[0]+(c_z[0]-c_y[0]-c_x[0])*200)),
                    np.hstack((c_eye[2], c_eye[2]+(c_z[2]-c_y[2]-c_x[2])*200)),
                    np.hstack((c_eye[1], c_eye[1]+(c_z[1]-c_y[1]-c_x[1])*200)),
                    ls='-', color='k')

        # export as pictures
        # plt.savefig(epoch_out_dir + '_'+str(i)+'.jpg')
        # continue

        # export as videos
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        rgb_image = image[:, :, :3]
        r,g,b=cv2.split(rgb_image)
        img_bgr = cv2.merge([b,g,r])
        videoWriter.write(img_bgr)
        plt.clf()
    videoWriter.release()
    videoWriter = None
    if sound:
        video_audio_path = os.path.join(epoch_out_dir, 'merge_' + video_name)
        video_add_audio(video_path, audio_path, video_audio_path)
        os.remove(video_path)
        
def CameraSave(camera_eye, camera_z, camera_y, camera_x, camera_rot, camera_fov, epoch, out_dir, camera_name):
    epoch_out_dir = os.path.join(out_dir, epoch)
    Path(epoch_out_dir).mkdir(parents=True, exist_ok=True)
    camera_path = os.path.join(epoch_out_dir, camera_name)
    with open(camera_path, 'w') as cf:
        json.dump({"camera_eye":camera_eye.tolist(),
        "camera_z":camera_z.tolist(),
        "camera_y":camera_y.tolist(),
        "camera_x":camera_x.tolist(),
        "camera_rot":camera_rot.tolist(),
        "Fov":camera_fov.tolist(),
        },cf)

def BoneMaskSave(bone_mask, epoch, out_dir, bonemask_name):
    epoch_out_dir = os.path.join(out_dir, epoch)
    Path(epoch_out_dir).mkdir(parents=True, exist_ok=True)
    bonemask_path = os.path.join(epoch_out_dir, bonemask_name)
    with open(bonemask_path, 'w') as bmf:
        json.dump({"bone_mask":bone_mask.tolist()
        },bmf)
def MotionCameraRenderSave(pose, camera_eye, camera_z, camera_y, camera_x, camera_rot, camera_fov, bone_mask, epoch,  out_dir, audio_path, camera_name, render_videos=True, sound=True):
    assert pose.shape[0] == camera_eye.shape[0]

    pose = np.array(pose, dtype='float32')
    camera_eye = np.array(camera_eye, dtype='float32')
    camera_z = np.array(camera_z, dtype='float32')
    camera_y = np.array(camera_y, dtype='float32')
    camera_x = np.array(camera_x, dtype='float32')
    camera_rot = np.array(camera_rot, dtype='float32')
    camera_fov = np.array(camera_fov, dtype='float32')
    bone_mask = np.array(bone_mask, dtype='float32')
    if render_videos:
        MotionCameraRender(pose.copy(), camera_eye.copy(), camera_z.copy(), camera_y.copy(), camera_x.copy(), camera_fov.copy(), bone_mask.copy(), epoch, out_dir, audio_path, sound)
    CameraSave(camera_eye.copy(), camera_z.copy(), camera_y.copy(), camera_x.copy(), camera_rot.copy(), camera_fov.copy(), epoch, out_dir, camera_name)
    bonemask_name = 'bm'+camera_name[1:]
    BoneMaskSave(bone_mask.copy(), epoch, out_dir, bonemask_name)
    