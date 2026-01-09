# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from protenix.model.utils import centre_random_augmentation
from protenix.metrics.rmsd import weighted_rigid_align
import numpy as np
import copy
from torch.nn.functional import one_hot



class TokenNoiser:
    def __init__(
        self, 
        noise_token_id=20,
        timesteps=200, 
        discrete_schedule="linear",
        noise_type="discrete",
        continuous_factor=0.25,
        continuous_threshold=3.0
    ):
        self.timesteps = timesteps
        self.noise_token_id = noise_token_id
        if discrete_schedule == "linear":
            self.mask_rates = np.linspace(
                0, 1, timesteps, dtype=np.float64
            )
        elif discrete_schedule == "cosine":
            self.mask_rates = 1 - (np.cos(
                np.pi * np.linspace(0, 1, timesteps, dtype=np.float64)
            ) + 1.0) / 2
        self.noise_type=noise_type
        self.continuous_factor = continuous_factor
        self.continuous_threshold = continuous_threshold
    
    def discrete_corrupt(self, seq, times, corrupt_mask=None):
        mask_nums = torch.rand_like(seq, dtype=torch.float32)
        mask = torch.zeros_like(mask_nums, dtype=torch.bool)
        for i, t in enumerate(times):
            mask[i] = mask_nums[i] < self.mask_rates[t]
        
        if corrupt_mask is not None:
            mask = (mask * corrupt_mask).bool()

        res = copy.deepcopy(seq)
        for i, t in enumerate(times):
            corrupt_ids = self.noise_token_id * torch.ones_like(seq[i])
            res[i] = torch.where(mask[i], corrupt_ids, res[i])

        return res, mask
    
    def continuous_corrupt(self, seq, sigma, corrupt_mask=None):
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=32) # same to restype feature dimension
        else:
            seqs = seq
        seq_noise = torch.randn_like(seqs.float())
        if corrupt_mask is not None:
            seq_noise = seq_noise * corrupt_mask.unsqueeze(-1)
        res = seqs + self.continuous_factor * sigma * seq_noise
        res = torch.clamp(res, min=-self.continuous_threshold, max=self.continuous_threshold)
        return res

    def corrupt(self, seq, noise, corrupt_mask=None):
        if self.noise_type == "discrete":
            return self.discrete_corrupt(seq, noise, corrupt_mask)
        else:
            return self.continuous_corrupt(seq, noise, corrupt_mask)


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    token_noiser: Optional["TokenNoiser"] = None,
    use_sequence: bool = False,
    temperature: float = 1.0,
    inpaint: bool = False,
    coords_gt: Optional[torch.Tensor] = None,
    coords_mask: Optional[torch.Tensor] = None,
    resolved_atom_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        inpaint (bool): Whether to use structure inpainting. Defaults to False.
        coords_gt (Optional[torch.Tensor]): Ground truth coordinates for inpainting.
            [..., N_atom, 3]
        coords_mask (Optional[torch.Tensor]): Mask for atoms to design (True=design, False=keep GT).
            [..., N_atom]
        resolved_atom_mask (Optional[torch.Tensor]): Mask for resolved atoms used in alignment.
            [..., N_atom]

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    N_step = len(noise_schedule) - 1

    seq_denoised_final = None
    if use_sequence and token_noiser is not None:
        token_gt = input_feature_dict.get("token_gt")
        cdr_mask = input_feature_dict.get("cdr_mask")
        if token_gt is not None:
            token_gt = token_gt.unsqueeze(0).expand(N_sample, -1)
            input_feature_dict["token_gt"] = token_gt
        if cdr_mask is not None:
            cdr_mask = cdr_mask.unsqueeze(0).expand(N_sample, -1)
            input_feature_dict["cdr_mask"] = cdr_mask
        if "attn_mask" in input_feature_dict:
            input_feature_dict["attn_mask"] = input_feature_dict["attn_mask"].unsqueeze(0).expand(N_sample, -1)

        if token_noiser.noise_type == "discrete":
            init_t = torch.tensor([N_step - 1] * N_sample, device=device)
            seq_noisy, _ = token_noiser.corrupt(token_gt, init_t, cdr_mask)
        else:
            init_sigma = noise_schedule[0]
            seq_noisy = token_noiser.corrupt(token_gt, init_sigma, cdr_mask)
        input_feature_dict["token_noisy"] = seq_noisy

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        seq_denoised_final = None
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )

        for step_idx, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised, s_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            if inpaint and coords_gt is not None:
                atom_weight = torch.ones(x_denoised.shape[:-1], device=device, dtype=dtype)
                if resolved_atom_mask is not None:
                    atom_weight = atom_weight * resolved_atom_mask.float()
                with torch.autocast("cuda", enabled=False):
                    coords_gt_aligned = weighted_rigid_align(
                        coords_gt.float().expand_as(x_denoised),
                        x_denoised.float(),
                        atom_weight.float(),
                        stop_gradient=True,
                    )
                coords_gt_aligned = coords_gt_aligned.to(x_denoised.dtype)
                x_denoised = torch.where(
                    coords_mask[..., None].bool(), x_denoised, coords_gt_aligned
                )

            if use_sequence and token_noiser is not None and s_denoised is not None:
                t = N_step - 1 - step_idx
                if t > 0:
                    token_gt = input_feature_dict.get("token_gt")
                    cdr_mask = input_feature_dict.get("cdr_mask")
                    seq_mask = input_feature_dict.get("seq_mask", cdr_mask)
                    
                    if token_noiser.noise_type == "discrete":
                        seq_pred = Categorical(logits=s_denoised * temperature).sample()
                        next_t = torch.tensor([t - 1] * chunk_n_sample, device=device)
                        seq_noisy, _ = token_noiser.corrupt(seq_pred, next_t, seq_mask)
                        if token_gt is not None and cdr_mask is not None:
                            noise_gt, _ = token_noiser.corrupt(token_gt, next_t, cdr_mask)
                            seq_noisy = torch.where(seq_mask.bool(), seq_noisy, noise_gt)
                    else:
                        seq_prob = F.softmax(s_denoised * temperature, dim=-1)
                        next_sigma = c_tau * (1 + gamma0 if c_tau > gamma_min else 1)
                        seq_noisy = token_noiser.corrupt(seq_prob, next_sigma, seq_mask)
                        if token_gt is not None and cdr_mask is not None:
                            noise_gt = token_noiser.corrupt(token_gt, next_sigma, cdr_mask)
                            seq_noisy = torch.where(seq_mask.unsqueeze(-1).bool(), seq_noisy, noise_gt)
                    
                    input_feature_dict["token_noisy"] = seq_noisy
                else:
                    seq_denoised_final = s_denoised

            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        if inpaint and coords_gt is not None:
            atom_weight = torch.ones(x_l.shape[:-1], device=device, dtype=dtype)
            if resolved_atom_mask is not None:
                atom_weight = atom_weight * resolved_atom_mask.float()
            with torch.autocast("cuda", enabled=False):
                coords_gt_aligned = weighted_rigid_align(
                    coords_gt.float().expand_as(x_l),
                    x_l.float(),
                    atom_weight.float(),
                    stop_gradient=True,
                )
            coords_gt_aligned = coords_gt_aligned.to(x_l.dtype)
            x_l = torch.where(coords_mask[..., None].bool(), x_l, coords_gt_aligned)

        return x_l, seq_denoised_final

    if diffusion_chunk_size is None:
        x_l, seq_denoised_final = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        seq_chunks = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l, chunk_seq = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
            if chunk_seq is not None:
                seq_chunks.append(chunk_seq)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
        if len(seq_chunks) > 0:
            seq_denoised_final = torch.cat(seq_chunks, dim=-3)
    
    if use_sequence:
        return x_l, seq_denoised_final
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    token_noiser: Optional[TokenNoiser],
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    input_feature_dict["token_gt"] = input_feature_dict["token_gt"].unsqueeze(0).expand(N_sample, -1)
    input_feature_dict["cdr_mask"] = input_feature_dict["cdr_mask"].unsqueeze(0).expand(N_sample, -1)
    if "attn_mask" in input_feature_dict:
        input_feature_dict["attn_mask"] = input_feature_dict["attn_mask"].unsqueeze(0).expand(N_sample, -1)
    
    if token_noiser is not None:
        if token_noiser.noise_type == "discrete":
            times = torch.randint(token_noiser.timesteps, size=(*batch_size_shape, N_sample), device=device)
            sigma = input_feature_dict["noise_schedule"]
            gamma = torch.where(sigma > input_feature_dict["gamma_min"], input_feature_dict["gamma0"], 0.0)
            sigma = sigma[:-1] * (1 +  gamma[1:])
            sigma = sigma[token_noiser.timesteps - 1 - times]
            input_feature_dict["token_noisy"], input_feature_dict["token_mask"] = \
            token_noiser.corrupt(input_feature_dict["token_gt"], times, input_feature_dict["cdr_mask"])
        else:
            # Add independent noise to each structure
            # sigma: independent noise-level [..., N_sample]
            sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
            # noise: [..., N_sample, N_atom, 3]
            input_feature_dict["token_noisy"] = \
            token_noiser.corrupt(input_feature_dict["token_gt"], sigma, input_feature_dict["cdr_mask"])

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    # sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised, s_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        x_denoised, s_denoised = [], []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i, s_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                use_conditioning=use_conditioning,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            x_denoised.append(x_denoised_i)
            if s_denoised_i is not None:
                s_denoised.append(s_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)
        if len(s_denoised) > 0:
            s_denoised = torch.cat(s_denoised, dim=-3)
        else:
            s_denoised = None

    return x_gt_augment, x_denoised, sigma, s_denoised
