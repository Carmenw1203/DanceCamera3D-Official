import multiprocessing
import os
import sys
sys.path.append("..")
sys.path.append(".")
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.amc_dataset import AMCDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.diffusion_SepCFG import GaussianDiffusion_SepCFG
from model.model import DanceCameraDecoder, DanceCameraDecoder_SepCFG

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class DanceCamera3D:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer_pose= None,
        normalizer_camera_dis = None,
        normalizer_camera_pos = None,
        normalizer_camera_rot = None,
        normalizer_camera_fov = None,
        normalizer_camera_eye = None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        camera_format='polar',
        condition_separation_CFG = True,
        gw1 = 1,
        gw2 = 2,
        w_loss=2,
        w_v_loss=5,
        w_a_loss=5,
        w_in_ba_loss=0.0015,
        w_out_ba_loss=0,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"

        self.motion_repr_dim = motion_repr_dim = 60 * 3 # 60 joints, keypoints3d
        self.camera_format = camera_format
        if camera_format == 'polar':
            self.camera_repr_dim = camera_repr_dim = 1+ 3 * 2 + 1 # dis, pos, rot, fov
        elif camera_format == 'centric':
            self.camera_repr_dim = camera_repr_dim = 3 + 1 + 3 # rot, fov, eye

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            self.normalizer_pose = checkpoint["normalizer_pose"]
            self.normalizer_camera_dis = checkpoint["normalizer_camera_dis"]
            self.normalizer_camera_pos = checkpoint["normalizer_camera_pos"]
            self.normalizer_camera_rot = checkpoint["normalizer_camera_rot"]
            self.normalizer_camera_fov = checkpoint["normalizer_camera_fov"]
            if camera_format == 'centric':
                self.normalizer_camera_eye = checkpoint["normalizer_camera_eye"]
            else:
                self.normalizer_camera_eye = None

        if condition_separation_CFG:
            model = DanceCameraDecoder_SepCFG(
                nfeats=camera_repr_dim,
                seq_len=horizon,
                latent_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                m_cond_feature_dim=feature_dim,
                p_cond_dim=motion_repr_dim,
                activation=F.gelu,
            )
            diffusion = GaussianDiffusion_SepCFG(
                model,
                horizon,
                camera_repr_dim,
                schedule="cosine",
                n_timestep=1000,
                predict_epsilon=False,
                loss_type="l2",
                use_p2=False,
                p_cond_drop_prob=0.25,
                m_cond_drop_prob=0.25,
                guidance_weight1=gw1,
                guidance_weight2=gw2,
                w_loss=w_loss,
                w_v_loss=w_v_loss,
                w_a_loss=w_a_loss,
                w_in_ba_loss=w_in_ba_loss,
                w_out_ba_loss=w_out_ba_loss,
            )
        else:
            model = DanceCameraDecoder(
                nfeats=camera_repr_dim,
                seq_len=horizon,
                latent_dim=512,
                ff_size=1024,
                num_layers=8,
                num_heads=8,
                dropout=0.1,
                m_cond_feature_dim=feature_dim,
                p_cond_dim=motion_repr_dim,
                activation=F.gelu,
            )
            diffusion = GaussianDiffusion(
                model,
                horizon,
                camera_repr_dim,
                schedule="cosine",
                n_timestep=1000,
                predict_epsilon=False,
                loss_type="l2",
                use_p2=False,
                cond_drop_prob=0.25,
                guidance_weight=gw1,
                w_loss=w_loss,
                w_v_loss=w_v_loss,
                w_a_loss=w_a_loss,
                w_in_ba_loss=w_in_ba_loss,
                w_out_ba_loss=w_out_ba_loss,
            )

        
        
        

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
            not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AMCDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AMCDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer_pose= train_dataset.normalizer_pose,
                normalizer_camera_dis= train_dataset.normalizer_camera_dis,
                normalizer_camera_pos= train_dataset.normalizer_camera_pos,
                normalizer_camera_rot= train_dataset.normalizer_camera_rot,
                normalizer_camera_fov= train_dataset.normalizer_camera_fov,
                normalizer_camera_eye= train_dataset.normalizer_camera_eye,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer_pose = test_dataset.normalizer_pose
        self.normalizer_camera_dis = test_dataset.normalizer_camera_dis
        self.normalizer_camera_pos = test_dataset.normalizer_camera_pos
        self.normalizer_camera_rot = test_dataset.normalizer_camera_rot
        self.normalizer_camera_fov = test_dataset.normalizer_camera_fov
        self.normalizer_camera_eye = test_dataset.normalizer_camera_eye

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        # print("load_loop")
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        # print("accelerator.wait_for_everyone")
        for epoch in tqdm(range(1, opt.epochs + 1)):
            avg_loss = 0
            avg_vloss = 0
            avg_aloss = 0
            avg_in_ba_loss = 0
            avg_out_ba_loss = 0
            # train
            self.train()
            for step, (x_camera, x_bone_mask, pose_cond, music_cond, filename, wavnames) in enumerate(
                load_loop(train_data_loader)
            ):
                # print(step)
                total_loss, (loss, v_loss, a_loss, in_ba_loss, out_ba_loss) = self.diffusion(
                    x_camera, x_bone_mask, pose_cond, music_cond, self.normalizer_pose, self.normalizer_camera_dis, self.normalizer_camera_pos, self.normalizer_camera_rot, self.normalizer_camera_fov, self.normalizer_camera_eye, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_aloss += a_loss.detach().cpu().numpy()
                    avg_in_ba_loss += in_ba_loss.detach().cpu().numpy()
                    avg_out_ba_loss += out_ba_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
                    #log
                    log_dict = {
                        "Train Loss": avg_loss/ len(train_data_loader),
                        "V Loss": avg_vloss/ len(train_data_loader),
                        "A Loss": avg_aloss/ len(train_data_loader),
                        "InsideBodyAttention Loss": avg_in_ba_loss/ len(train_data_loader),
                        "OutsideBodyAttention Loss": avg_out_ba_loss/ len(train_data_loader),
                    }
                    wandb.log(log_dict)
        # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer_pose": self.normalizer_pose,
                        "normalizer_camera_dis": self.normalizer_camera_dis,
                        "normalizer_camera_pos": self.normalizer_camera_pos,
                        "normalizer_camera_rot": self.normalizer_camera_rot,
                        "normalizer_camera_fov": self.normalizer_camera_fov,
                        "normalizer_camera_eye": self.normalizer_camera_eye,
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = 2
                    shape = (render_count, self.horizon, self.camera_repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x_camera, x_bone_mask, pose_cond, music_cond, filename, wavnames) = next(iter(test_data_loader))
                    pose_cond = pose_cond.to(self.accelerator.device)
                    music_cond = music_cond.to(self.accelerator.device)
                    
                    self.diffusion.render_sample(
                        shape,
                        pose_cond[:render_count],
                        music_cond[:render_count],
                        self.normalizer_pose,
                        self.normalizer_camera_dis,
                        self.normalizer_camera_pos,
                        self.normalizer_camera_rot,
                        self.normalizer_camera_fov,
                        self.normalizer_camera_eye,
                        epoch,
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=wavnames[:render_count],
                        render_videos=opt.render_videos,
                        sound=True,
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process:
            wandb.run.finish()

    def render_sample(
        self, pose_cond, music_cond, wavnames, label, render_out, render_count=-1, render_videos=True, test_mode = 'normal', constraint = None
    ):
        assert len(pose_cond.shape) == len(music_cond.shape) == 3
        if render_count < 0:
            render_count = len(pose_cond)
        shape = (render_count, self.horizon, self.camera_repr_dim)
        pose_cond = self.normalizer_pose.normalize(pose_cond).to(self.accelerator.device)
        music_cond = music_cond.to(self.accelerator.device)
        print("render_sample")
        self.diffusion.render_sample(
            shape,
            pose_cond[:render_count],
            music_cond[:render_count],
            self.normalizer_pose,
            self.normalizer_camera_dis,
            self.normalizer_camera_pos,
            self.normalizer_camera_rot,
            self.normalizer_camera_fov,
            self.normalizer_camera_eye,
            label,
            render_out = render_out,
            name=wavnames[:render_count],
            render_videos=render_videos,
            sound=True,
            mode=test_mode,
            constraint = constraint
        )
