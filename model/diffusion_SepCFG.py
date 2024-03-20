import copy
import os
import glm
import math
import sys
import librosa as lr
sys.path.append("..")
import pickle
from pathlib import Path
from functools import partial
from my_utils.torch_glm import torch_glm_translate, torch_glm_rotate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from einops import reduce
from p_tqdm import p_map
from tqdm import tqdm


from .utils import extract, make_beta_schedule
from my_utils.vis import MotionCameraRenderSave

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion_SepCFG(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight1=3,
        guidance_weight2=3,
        use_p2=False,
        p_cond_drop_prob=0.2,
        m_cond_drop_prob=0.2,
        w_loss=2,
        w_v_loss=5,
        w_a_loss=5,
        w_in_ba_loss=0.0015,
        w_out_ba_loss=0,
    ):
        super().__init__()
        self.horizon = horizon
        self.repr_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

        self.p_cond_drop_prob = p_cond_drop_prob
        self.m_cond_drop_prob = m_cond_drop_prob
        self.w_loss = w_loss
        self.w_v_loss = w_v_loss
        self.w_a_loss = w_a_loss
        self.w_in_ba_loss = w_in_ba_loss
        self.w_out_ba_loss = w_out_ba_loss

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight1 = guidance_weight1
        self.guidance_weight2 = guidance_weight2

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, t, weight1=None, weight2=None, clip_x_start = False):
        weight1 = weight1 if weight1 is not None else self.guidance_weight1
        weight2 = weight2 if weight2 is not None else self.guidance_weight2
        model_output = self.model.guided_forward(x, cond, t, weight1, weight2)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight1 = min(self.guidance_weight1, 0)
            weight2 = min(self.guidance_weight2, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight1 = min(self.guidance_weight1, 1)
            weight2 = min(self.guidance_weight2, 1)
        else:
            weight1 = self.guidance_weight1
            weight2 = self.guidance_weight2

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight1, weight2)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x
    
    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        print("long_ddim_sample")
        print(cond.shape)
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
        
        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights1 = np.clip(np.linspace(0, self.guidance_weight1 * 2, sampling_timesteps), None, self.guidance_weight1)
        weights2 = np.clip(np.linspace(0, self.guidance_weight2 * 2, sampling_timesteps), None, self.guidance_weight2)
        time_pairs = list(zip(times[:-1], times[1:], weights1, weights2)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)
        
        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight1, weight2 in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight1=weight1, weight2=weight2, clip_x_start = self.clip_denoised) 

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, b_mask, p_cond, m_cond, normalizer_pose, normalizer_camera_dis, normalizer_camera_pos, normalizer_camera_rot, normalizer_camera_fov, normalizer_camera_eye, t):
        if self.repr_dim == 8:#polar coordinates
            x_start = x_start[:,:,:8]
        elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
            x_start = x_start[:,:,-7:]
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        pm_cond = torch.cat(
            [p_cond, m_cond],
            dim=-1,
        )
        x_recon = self.model(x_noisy, pm_cond, t, p_cond_drop_prob=self.p_cond_drop_prob, m_cond_drop_prob=self.m_cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # acceleration loss
        target_a = target_v[:, 1:] - target_v[:, :-1]
        model_out_a = model_out_v[:, 1:] - model_out_v[:, :-1]
        a_loss = self.loss_fn(model_out_a, target_a, reduction="none")
        a_loss = reduce(a_loss, "b ... -> b (...)", "mean")
        a_loss = a_loss * extract(self.p2_loss_weight, t, a_loss.shape)

        # body attention loss
        unnormalized_motion = normalizer_pose.unnormalize(p_cond)#(N, seq, joint*3)
        reshape_motion = unnormalized_motion.reshape(unnormalized_motion.shape[0], unnormalized_motion.shape[1], -1, 3)#(N, seq, joint, 3)
        transpose_motion = reshape_motion.transpose(2,1).transpose(0,1)#(joint, N, seq, 3)
        if self.repr_dim == 8:#polar coordinates
            out_camera_dis = normalizer_camera_dis.unnormalize(model_out[:,:,:1])#(N, seq, 1)
            out_camera_pos = normalizer_camera_pos.unnormalize(model_out[:,:,1:4])#(N, seq, 3)
            out_camera_rot = normalizer_camera_rot.unnormalize(model_out[:,:,4:7])#(N, seq, 3)
            out_camera_fov = normalizer_camera_fov.unnormalize(model_out[:,:,7:8])#(N, seq, 1)
        elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
            out_camera_rot = normalizer_camera_rot.unnormalize(model_out[:,:,:3])#(N, seq, 3)
            out_camera_fov = normalizer_camera_fov.unnormalize(model_out[:,:,3:4])#(N, seq, 1)
            out_camera_eye = normalizer_camera_eye.unnormalize(model_out[:,:,4:7])#(N, seq, 3)
        
        

        b, s, _ = out_camera_rot.shape
        device = out_camera_rot.device
        t1 = torch.ones(b,s,1).to(device)
        t0 = torch.zeros(b,s,1).to(device)
        view = torch.ones(b,s,4,4) * torch.eye(4)
        view = view.to(device)
        if self.repr_dim == 8:#polar coordinates
            camera_dis_expand = torch.cat([t0,t0, torch.abs(out_camera_dis)],dim=-1)
            view = torch_glm_translate(view,camera_dis_expand)
        rot = torch.ones(b,s,4,4) * torch.eye(4)
        rot = rot.to(device)
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,1],torch.cat([t0,t1,t0], dim=-1))
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,2],torch.cat([t0,t0,t1*(-1)], dim=-1))
        rot = torch_glm_rotate(rot,out_camera_rot[:,:,0],torch.cat([t1,t0,t0], dim=-1))
        view = torch.matmul(view,rot)

        if self.repr_dim == 8:#polar coordinates
            out_camera_eye = view[:,:,3,:3] + out_camera_pos * torch.cat([t1,t1,t1*(-1)], dim=-1)
        out_camera_z = F.normalize(view[:,:,2,:3]*(-1),p=2,dim=-1)
        out_camera_y = F.normalize(view[:,:,1,:3],p=2,dim=-1)
        out_camera_x = F.normalize(view[:,:,0,:3],p=2,dim=-1)

        out_motion2eye = transpose_motion - out_camera_eye
        out_kps_yz = out_motion2eye - out_camera_x * torch.sum(out_motion2eye * out_camera_x, axis=-1, keepdims = True) 
        out_kps_xz = out_motion2eye - out_camera_y * torch.sum(out_motion2eye * out_camera_y, axis=-1, keepdims = True) 
        out_cos_y_z = torch.sum(out_kps_yz * out_camera_z, axis=-1)
        out_cos_x_z = torch.sum(out_kps_xz * out_camera_z, axis=-1)
        out_cos_fov = torch.cos(out_camera_fov*0.5/180 * math.pi)
        out_cos_fov = out_cos_fov.reshape(out_cos_fov.shape[0], out_cos_fov.shape[1])#(N, seq)
        
        out_diff_x = (out_cos_fov * torch.sqrt(torch.sum(out_kps_xz * out_kps_xz, axis=-1)) - out_cos_x_z).transpose(0,1).transpose(2,1)#(N, seq, joint)
        out_diff_y = (out_cos_fov * torch.sqrt(torch.sum(out_kps_yz * out_kps_yz, axis=-1)) - out_cos_y_z).transpose(0,1).transpose(2,1)

        in_ba_loss = F.relu(out_diff_x*b_mask)+F.relu(out_diff_y*b_mask)# inside camera view 
        out_ba_loss = F.relu(out_diff_x*(b_mask-1)) * F.relu(out_diff_y*(b_mask-1))# outside camera view
        
        losses = (
            self.w_loss  * loss.mean(),
            self.w_v_loss * v_loss.mean(),
            self.w_a_loss * a_loss.mean(),
            self.w_in_ba_loss * in_ba_loss.mean(),
            self.w_out_ba_loss * out_ba_loss.mean(),
        )
        return sum(losses), losses

    def loss(self, x, b_mask, p_cond, m_cond, normalizer_pose, normalizer_camera_dis, normalizer_camera_pos, normalizer_camera_rot, normalizer_camera_fov, normalizer_camera_eye, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, b_mask, p_cond, m_cond, normalizer_pose, normalizer_camera_dis, normalizer_camera_pos, normalizer_camera_rot, normalizer_camera_fov, normalizer_camera_eye, t)

    def forward(self, x, b_mask, p_cond, m_cond, normalizer_pose, normalizer_camera_dis, normalizer_camera_pos, normalizer_camera_rot, normalizer_camera_fov, normalizer_camera_eye, t_override=None):
        return self.loss(x, b_mask, p_cond, m_cond, normalizer_pose, normalizer_camera_dis, normalizer_camera_pos, normalizer_camera_rot, normalizer_camera_fov, normalizer_camera_eye, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample(
        self,
        shape,
        p_cond,
        m_cond,
        normalizer_pose,
        normalizer_camera_dis,
        normalizer_camera_pos,
        normalizer_camera_rot,
        normalizer_camera_fov,
        normalizer_camera_eye,
        epoch,
        render_out,
        name=None,
        render_videos=True,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        start_point=None
    ):
        pm_cond = torch.cat(
                [p_cond, m_cond],
                dim=-1,
            )
        if isinstance(shape, tuple):
            if mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long" or mode == "long_no_interpolation":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    pm_cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape
        
        if self.repr_dim == 8:#polar coordinates
            samples_camera_dis = samples[:,:,:1]
            samples_camera_pos = samples[:,:,1:4]
            samples_camera_rot = samples[:,:,4:7]
            samples_camera_fov = samples[:,:,7:8]
            samples_camera_dis = normalizer_camera_dis.unnormalize(samples_camera_dis)
            samples_camera_pos = normalizer_camera_pos.unnormalize(samples_camera_pos)
            samples_camera_rot = normalizer_camera_rot.unnormalize(samples_camera_rot)
            samples_camera_fov = normalizer_camera_fov.unnormalize(samples_camera_fov)
        elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
            samples_camera_rot = samples[:,:,:3]
            samples_camera_fov = samples[:,:,3:4]
            samples_camera_eye = samples[:,:,4:7]
            samples_camera_rot = normalizer_camera_rot.unnormalize(samples_camera_rot)
            samples_camera_fov = normalizer_camera_fov.unnormalize(samples_camera_fov)
            samples_camera_eye = normalizer_camera_eye.unnormalize(samples_camera_eye)

        cond_pose = normalizer_pose.unnormalize(p_cond).detach().cpu()
        


        b, s, _ = samples.shape
        device = samples.device
        

        
        if mode == 'long' and b > 1:
            assert s % 2 == 0
            half = s // 2
            fade_out = torch.ones((1, s, 1)).to(device)
            fade_in = torch.ones((1, s, 1)).to(device)
            fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                device
            )
            fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                device
            )

            cond_pose[:-1] *= fade_out
            cond_pose[1:] *= fade_in
            if self.repr_dim == 8:#polar coordinates
                samples_camera_dis[:-1] *= fade_out
                samples_camera_dis[1:] *= fade_in
                samples_camera_pos[:-1] *= fade_out
                samples_camera_pos[1:] *= fade_in
                samples_camera_fov[:-1] *= fade_out
                samples_camera_fov[1:] *= fade_in
                samples_camera_rot[:-1] *= fade_out
                samples_camera_rot[1:] *= fade_in
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                samples_camera_fov[:-1] *= fade_out
                samples_camera_fov[1:] *= fade_in
                samples_camera_rot[:-1] *= fade_out
                samples_camera_rot[1:] *= fade_in
                samples_camera_eye[:-1] *= fade_out
                samples_camera_eye[1:] *= fade_in

            full_cond_pose = torch.zeros((s + half * (b - 1), cond_pose.shape[-1])).to(device)
            if self.repr_dim == 8:#polar coordinates
                full_camera_dis = torch.zeros((s + half * (b - 1), samples_camera_dis.shape[-1])).to(device)
                full_camera_pos = torch.zeros((s + half * (b - 1), samples_camera_pos.shape[-1])).to(device)
                full_camera_fov = torch.zeros((s + half * (b - 1), samples_camera_fov.shape[-1])).to(device)
                full_camera_rot = torch.zeros((s + half * (b - 1), samples_camera_rot.shape[-1])).to(device)
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                full_camera_fov = torch.zeros((s + half * (b - 1), samples_camera_fov.shape[-1])).to(device)
                full_camera_rot = torch.zeros((s + half * (b - 1), samples_camera_rot.shape[-1])).to(device)
                full_camera_eye = torch.zeros((s + half * (b - 1), samples_camera_eye.shape[-1])).to(device)

            idx = 0
            while(idx < b):
                full_cond_pose[idx*half : idx*half + s] += cond_pose[idx]
                if self.repr_dim == 8:#polar coordinates
                    full_camera_dis[idx*half : idx*half + s] += samples_camera_dis[idx]
                    full_camera_pos[idx*half : idx*half + s] += samples_camera_pos[idx]
                    full_camera_fov[idx*half : idx*half + s] += samples_camera_fov[idx]
                    full_camera_rot[idx*half : idx*half + s] += samples_camera_rot[idx]#
                elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                    full_camera_fov[idx*half : idx*half + s] += samples_camera_fov[idx]
                    full_camera_rot[idx*half : idx*half + s] += samples_camera_rot[idx]#
                    full_camera_eye[idx*half : idx*half + s] += samples_camera_eye[idx]#
                idx += 1

            full_cond_pose = full_cond_pose.unsqueeze(0)
            if self.repr_dim == 8:#polar coordinates
                full_camera_dis = full_camera_dis.unsqueeze(0)
                full_camera_pos = full_camera_pos.unsqueeze(0)
                full_camera_fov = full_camera_fov.unsqueeze(0)
                full_camera_rot = full_camera_rot.unsqueeze(0)
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                full_camera_fov = full_camera_fov.unsqueeze(0)
                full_camera_rot = full_camera_rot.unsqueeze(0)
                full_camera_eye = full_camera_eye.unsqueeze(0)
        elif mode == 'long_no_interpolation' and b > 1:
            assert s % 2 == 0
            half = s // 2
            full_cond_pose = torch.zeros((s + half * (b - 1), cond_pose.shape[-1])).to(device)
            if self.repr_dim == 8:#polar coordinates
                full_camera_dis = torch.zeros((s + half * (b - 1), samples_camera_dis.shape[-1])).to(device)
                full_camera_pos = torch.zeros((s + half * (b - 1), samples_camera_pos.shape[-1])).to(device)
                full_camera_fov = torch.zeros((s + half * (b - 1), samples_camera_fov.shape[-1])).to(device)
                full_camera_rot = torch.zeros((s + half * (b - 1), samples_camera_rot.shape[-1])).to(device)
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                full_camera_fov = torch.zeros((s + half * (b - 1), samples_camera_fov.shape[-1])).to(device)
                full_camera_rot = torch.zeros((s + half * (b - 1), samples_camera_rot.shape[-1])).to(device)
                full_camera_eye = torch.zeros((s + half * (b - 1), samples_camera_eye.shape[-1])).to(device)

            idx = 0
            while(idx < b):
                full_cond_pose[idx*half : idx*half + half] += cond_pose[idx][:half]
                if self.repr_dim == 8:#polar coordinates
                    full_camera_dis[idx*half : idx*half + half] += samples_camera_dis[idx][:half]
                    full_camera_pos[idx*half : idx*half + half] += samples_camera_pos[idx][:half]
                    full_camera_fov[idx*half : idx*half + half] += samples_camera_fov[idx][:half]
                    full_camera_rot[idx*half : idx*half + half] += samples_camera_rot[idx][:half]#
                elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                    full_camera_fov[idx*half : idx*half + half] += samples_camera_fov[idx][:half]
                    full_camera_rot[idx*half : idx*half + half] += samples_camera_rot[idx][:half]#
                    full_camera_eye[idx*half : idx*half + half] += samples_camera_eye[idx][:half]
                idx += 1

            full_cond_pose[-half:] += cond_pose[-1][-half:]
            full_cond_pose = full_cond_pose.unsqueeze(0)
            if self.repr_dim == 8:#polar coordinates
                full_camera_dis[-half:] += samples_camera_dis[-1][-half:]
                full_camera_pos[-half:] += samples_camera_pos[-1][-half:]
                full_camera_fov[-half:] += samples_camera_fov[-1][-half:]
                full_camera_rot[-half:] += samples_camera_rot[-1][-half:]

                full_camera_dis = full_camera_dis.unsqueeze(0)
                full_camera_pos = full_camera_pos.unsqueeze(0)
                full_camera_fov = full_camera_fov.unsqueeze(0)
                full_camera_rot = full_camera_rot.unsqueeze(0)
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                full_camera_fov[-half:] += samples_camera_fov[-1][-half:]
                full_camera_rot[-half:] += samples_camera_rot[-1][-half:]
                full_camera_eye[-half:] += samples_camera_eye[-1][-half:]

                full_camera_fov = full_camera_fov.unsqueeze(0)
                full_camera_rot = full_camera_rot.unsqueeze(0)
                full_camera_eye = full_camera_eye.unsqueeze(0)
        else:
            full_cond_pose = cond_pose
            if self.repr_dim == 8:#polar coordinates
                full_camera_dis = samples_camera_dis
                full_camera_pos = samples_camera_pos
                full_camera_fov = samples_camera_fov
                full_camera_rot = samples_camera_rot
            elif self.repr_dim == 7:#camera centric representation/ Cartesian coordinates
                full_camera_fov = samples_camera_fov
                full_camera_rot = samples_camera_rot
                full_camera_eye = samples_camera_eye
        
        b, s, _ = full_camera_rot.shape
        t1 = torch.ones(b,s,1).to(device)
        t0 = torch.zeros(b,s,1).to(device)
        
        view = torch.ones(b,s,4,4) * torch.eye(4)
        view = view.to(device)
        if self.repr_dim == 8:#polar coordinates
            full_camera_dis_expand = torch.cat([t0,t0,torch.abs(full_camera_dis)],dim=-1)
            view = torch_glm_translate(view,full_camera_dis_expand)
        rot = torch.ones(b,s,4,4) * torch.eye(4)
        rot = rot.to(device)
        rot = torch_glm_rotate(rot,full_camera_rot[:,:,1],torch.cat([t0,t1,t0], dim=-1))
        rot = torch_glm_rotate(rot,full_camera_rot[:,:,2],torch.cat([t0,t0,t1*(-1)], dim=-1))
        rot = torch_glm_rotate(rot,full_camera_rot[:,:,0],torch.cat([t1,t0,t0], dim=-1))

        view = torch.matmul(view,rot)
        if self.repr_dim == 8:#polar coordinates
            full_camera_eye = view[:,:,3,:3] + full_camera_pos * torch.cat([t1,t1,t1*(-1)], dim=-1)
        full_camera_z = F.normalize(view[:,:,2,:3]*(-1),p=2,dim=-1)
        full_camera_y = F.normalize(view[:,:,1,:3],p=2,dim=-1)
        full_camera_x = F.normalize(view[:,:,0,:3],p=2,dim=-1)
        
        unnormalized_motion = normalizer_pose.unnormalize(p_cond).detach().cpu()#(N, seq, joint*3)
        reshape_motion = full_cond_pose.reshape(full_cond_pose.shape[0], full_cond_pose.shape[1], -1, 3)#(N, seq, joint, 3)
        transpose_motion = reshape_motion.transpose(2,1).transpose(0,1)#(joint, N, seq, 3)

        full_motion2eye = transpose_motion - full_camera_eye
        full_kps_yz = full_motion2eye - full_camera_x * torch.sum(full_motion2eye * full_camera_x, axis=-1, keepdims = True) 
        full_kps_xz = full_motion2eye - full_camera_y * torch.sum(full_motion2eye * full_camera_y, axis=-1, keepdims = True) 
        full_cos_y_z = torch.sum(full_kps_yz * full_camera_z, axis=-1)
        full_cos_x_z = torch.sum(full_kps_xz * full_camera_z, axis=-1)
        full_cos_fov = torch.cos(full_camera_fov*0.5/180 * math.pi)
        full_cos_fov = full_cos_fov.reshape(full_cos_fov.shape[0], full_cos_fov.shape[1])#(N, seq)

        full_diff_x = (full_cos_x_z - full_cos_fov * torch.sqrt(torch.sum(full_kps_xz * full_kps_xz, axis=-1))).transpose(0,1).transpose(2,1)#(N, seq, joint)
        full_diff_y = (full_cos_y_z - full_cos_fov * torch.sqrt(torch.sum(full_kps_yz * full_kps_yz, axis=-1))).transpose(0,1).transpose(2,1)

        full_diff_x[full_diff_x >= 0] = 1
        full_diff_x[full_diff_x < 0] = 0 
        full_diff_y[full_diff_y >= 0] = 1
        full_diff_y[full_diff_y < 0] = 0
        
        full_bone_mask = full_diff_x + full_diff_y
        full_bone_mask[full_bone_mask < 2] = 0
        full_bone_mask[full_bone_mask >= 2] = 1

        def inner(xx):
            num, (pose, camera_eye, camera_z, camera_y, camera_x, camera_rot, camera_fov, bone_mask) = xx
            filename = name[num] if name is not None else None
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            # print(pathparts[-1])
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            
            # path is like "DCM_data/amc_data_split_by_categories/Train/{features}/name"
            pathparts[-2] = "Audio_sliced"
            audio_path = os.path.join(*pathparts)
            camera_name = 'c'+pathparts[-1].split('slice')[0][1:-2]+pathparts[-1].split('slice')[1][:-4]+'.json'
            
            MotionCameraRenderSave(
                pose,
                camera_eye,
                camera_z,
                camera_y,
                camera_x,
                camera_rot,
                camera_fov,
                bone_mask,
                epoch=f"e{epoch}_short",
                out_dir=render_out,
                audio_path=audio_path,
                camera_name=camera_name,
                render_videos=render_videos,
                sound=sound
            )
        if mode == 'long' or mode == 'long_no_interpolation':
            # stitch audio
            audio, sr = lr.load(name[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n in name[1:]:
                audio, sr = lr.load(n, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            filename = name[0]
            path = os.path.normpath(filename)
            # path is like "DCM_cache_feature/songname/Audio_LongSliced/xxxx_slicen.wav"
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("_slice", "_stitch")
            pathparts[-2] = 'Audio_Stitch'
            
            audio_path = os.path.join(*pathparts)
            sf.write(audio_path, total_wav, sr)
            camera_name = 'c'+pathparts[-1].split('stitch')[0][1:-2]+pathparts[-1].split('stitch')[1][:-4]+'.json'
            # render
            MotionCameraRenderSave(
                full_cond_pose[0],
                full_camera_eye[0],
                full_camera_z[0],
                full_camera_y[0],
                full_camera_x[0],
                full_camera_rot[0],
                full_camera_fov[0],
                full_bone_mask[0],
                epoch=f"{mode}-{epoch}",
                out_dir=render_out,
                audio_path=audio_path,
                camera_name=camera_name,
                render_videos=render_videos,
                sound=sound
            )
        else:
            p_map(inner, enumerate(zip(full_cond_pose,
                                    full_camera_eye,
                                    full_camera_z,
                                    full_camera_y,
                                    full_camera_x,
                                    full_camera_rot,
                                    full_camera_fov,
                                    full_bone_mask)))
