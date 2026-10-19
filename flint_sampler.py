"""FLINT solver on Stable Diffusion 3.5-Medium.

Stage 1  Proximal source initialization: the mean mu of a variational initial noise
         q(x1) = N(mu, rho^2 I) is optimized with the manifold-constrained proximal objective
         (spectrally weighted data consistency on a K-step truncated trajectory, KL to N(0, I)
         and a cumulant regularizer).
Stage 2  Posterior sampling with bounded stochasticity: guided Euler sampling from mu with
         the stochasticity schedule eta(t) = lambda_s * sqrt(1 - sigma_t) * sigma_t.
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderTiny, StableDiffusion3Pipeline


def _autocast(dtype=torch.float16, enabled=True):
    return torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled)


# ---------------------------------------------------------------------------
# Stage 1 building blocks
# ---------------------------------------------------------------------------
class VariationalNoise(nn.Module):
    """q(x1) = N(mu, rho^2 I); rho is annealed linearly from sigma_max to sigma_min."""

    def __init__(self, noise_shape, sigma_max, sigma_min):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.dim = int(np.prod(noise_shape))
        self.mu = nn.Parameter(torch.randn(noise_shape))

    def rho(self, step, total_steps):
        p = step / max(total_steps - 1, 1)
        return self.sigma_max + (self.sigma_min - self.sigma_max) * p

    def sample(self, step, total_steps):
        return self.mu + self.rho(step, total_steps) * torch.randn_like(self.mu)

    def kl_divergence(self, step, total_steps):
        rho = self.rho(step, total_steps)
        D = float(self.dim)
        return 0.5 * (self.mu.flatten().pow(2).sum() + D * rho**2 - D - D * math.log(rho**2 + 1e-9))


def cumulant_regularizer(mu):
    """Per-channel Edgeworth approximation of KL(p_mu || N(0, 1)):
    Gaussian KL + skewness^2 / 12 + excess-kurtosis^2 / 48."""
    B, C, H, W = mu.shape
    losses = []
    for c in range(C):
        x = mu[0, c].flatten()
        mean = x.mean()
        cent = x - mean
        m2 = cent.pow(2).mean()
        var_c = m2.clamp(min=1e-8)
        m3 = cent.pow(3).mean()
        m4 = cent.pow(4).mean()
        gamma1 = m3 / (m2.pow(1.5) + 1e-8)
        gamma2 = m4 / (m2.pow(2) + 1e-8) - 3.0
        losses.append(0.5 * (mean.pow(2) + var_c - 1.0 - torch.log(var_c))
                      + (1.0 / 12.0) * gamma1.pow(2)
                      + (1.0 / 48.0) * gamma2.pow(2))
    return torch.stack(losses).mean()


def wiener_mask(t_stop, H, W, device, alpha=2.0, omega_c=0.05, W_cut=0.5, eps=1e-8):
    """SWDC weight W(omega, t_stop) for a 1/|omega|^alpha prior, rescaled so that
    W(omega_c) = W_cut (half-power point) and clipped to [0, 1]."""
    freq_h = torch.fft.fftfreq(H, device=device)
    freq_w = torch.fft.rfftfreq(W, device=device)
    gh, gw = torch.meshgrid(freq_h, freq_w, indexing='ij')
    omega_sq = (gh ** 2 + gw ** 2).clamp(min=eps)
    radius = torch.sqrt(omega_sq)

    S_x0 = 1.0 / (omega_sq ** (alpha / 2.0))
    S_x0 = S_x0 / S_x0.max()
    signal = (1.0 - t_stop) ** 2 * S_x0
    W_raw = signal / (signal + t_stop ** 2 + eps)

    ring_tol = (radius.max() - radius.min()) / min(H, W)
    ring = (radius - omega_c).abs() <= ring_tol
    if ring.any():
        W_raw_at_cutoff = W_raw[ring].mean()
    else:
        W_raw_at_cutoff = W_raw.flatten()[(radius - omega_c).abs().flatten().argmin()]
    W_raw_at_cutoff = W_raw_at_cutoff.clamp(min=eps)
    return (W_raw * (W_cut / W_raw_at_cutoff)).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def apply_mask(x, mask):
    H, W = x.shape[-2], x.shape[-1]
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * mask, s=(H, W), dim=(-2, -1))


# ---------------------------------------------------------------------------
# FLINT
# ---------------------------------------------------------------------------
class FLINT:
    def __init__(self, model_key='stabilityai/stable-diffusion-3.5-medium', device='cuda',
                 dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=dtype)
        self.scheduler = pipe.scheduler
        self.text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
        # Tiny SD3 autoencoder (TAESD3) as decoder, as in all experiments of the paper.
        self.vae = AutoencoderTiny.from_pretrained('madebyollin/taesd3', torch_dtype=dtype).to(device).eval()
        self.transformer = pipe.transformer.to(device).eval()
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        pipe.transformer, pipe.vae = None, None
        self._pipe = pipe
        self.vae_scale_factor = 8

    # ---- text conditioning -------------------------------------------------
    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]):
        """Encode the prompts (and the null prompt) once, then free the text encoders."""
        for enc in self.text_encoders:
            enc.to(self.device)
        embs = {p: self._encode(p) for p in dict.fromkeys(prompts)}
        null = self._encode([''])
        for enc in self.text_encoders:
            enc.to('cpu')
        del self.text_encoders, self._pipe
        torch.cuda.empty_cache()
        return embs, null

    def _encode(self, prompt):
        emb, _, pooled, _ = self._pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt, device=torch.device(self.device),
            num_images_per_prompt=1, do_classifier_free_guidance=False, max_sequence_length=256)
        return emb.to(self.dtype), pooled.to(self.dtype)

    # ---- model helpers -----------------------------------------------------
    def decode(self, z):
        return self.vae.decode(z / self.vae.config.scaling_factor + self.vae.config.shift_factor,
                               return_dict=False)[0]

    def velocity(self, z, t, emb, null, cfg_scale):
        v = self.transformer(hidden_states=z, timestep=t, pooled_projections=emb[1],
                             encoder_hidden_states=emb[0], return_dict=False)[0]
        if cfg_scale != 1.0:
            v_null = self.transformer(hidden_states=z, timestep=t, pooled_projections=null[1],
                                      encoder_hidden_states=null[0], return_dict=False)[0]
            v = v_null + cfg_scale * (v - v_null)
        return v

    def schedule(self, num_steps):
        """Timesteps / noise levels of the SD3.5 Euler schedule with `num_steps` steps."""
        self.scheduler.set_timesteps(num_steps + 1, device=self.device)
        timesteps = self.scheduler.timesteps
        return timesteps, timesteps.float() / self.scheduler.config.num_train_timesteps

    # ---- Stage 1 -----------------------------------------------------------
    def truncated_sample(self, x1, timesteps, sigmas, emb, null, cfg_scale):
        """K-step truncated Euler trajectory; only the first velocity keeps the graph (SDO shortcut)."""
        z = x1.clone().to(dtype=self.dtype)
        t_buf = torch.empty(z.shape[0], device=self.device, dtype=timesteps.dtype)
        for i in range(len(timesteps) - 1):
            t_buf.fill_(timesteps[i])
            with torch.set_grad_enabled(i == 0):
                v = self.velocity(z, t_buf, emb, null, cfg_scale)
            z = z + v * (sigmas[i + 1] - sigmas[i])
        return z

    def swdc_loss(self, z0, A, y, t_stop, sigma_y, omega_c, W_cut, alpha, eta_tilde=1.0, phi=1.0):
        """Spectrally weighted data consistency (Eq. 8) on the back-projected residual."""
        eta = max(1e-4, sigma_y ** 2 * eta_tilde)
        w = 1 - ((1 - t_stop.detach().clone().to(self.device, torch.float32)).clamp(0, 1)) ** phi

        x = self.decode(z0).float()
        B, C, H, W = x.shape
        with torch.autocast(device_type='cuda', enabled=False):
            Ax = A.A(x.reshape(B, -1).contiguous().float()).float()
            bp_x = A.A_pinv_add_eta(Ax.float(), eta).reshape(B, C, H, W).float()
            bp_y = A.A_pinv_add_eta(y.reshape(y.size(0), -1).contiguous().float(), eta).reshape(B, C, H, W).float()

        mask = wiener_mask(t_stop, H, W, self.device, alpha=alpha, omega_c=omega_c, W_cut=W_cut)
        residual = apply_mask(bp_x, mask) - apply_mask(bp_y, mask)
        return w * torch.mean(residual ** 2)

    def stage1(self, y, A, emb, null, cfg):
        """Proximal source initialization. Returns the optimized mean mu (the initial noise)."""
        H = cfg['img_size'] // self.vae_scale_factor
        noise_shape = (1, self.transformer.config.in_channels, H, H)
        q = VariationalNoise(noise_shape, cfg['sigma_max'], cfg['sigma_min']).to(self.device)
        optimizer = torch.optim.Adam([q.mu], lr=cfg['lr'])
        scaler = torch.amp.GradScaler('cuda', init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5,
                                      growth_interval=100)

        # K truncated steps taken from the start of a `stage1_ref_steps`-step schedule.
        timesteps, _ = self.schedule(cfg['stage1_ref_steps'])
        inner = timesteps[:cfg['K'] + 1].clone()
        inner[-1] = timesteps[-1]
        inner_sigmas = inner.float() / self.scheduler.config.num_train_timesteps
        t_stop = inner_sigmas[-2]

        for it in range(cfg['iters']):
            optimizer.zero_grad()
            x1 = q.sample(it, cfg['iters'])
            with _autocast(self.dtype):
                z0 = self.truncated_sample(x1, inner, inner_sigmas, emb, null, cfg['cfg_scale'])
                loss_dc = self.swdc_loss(z0, A, y, t_stop, cfg['sigma_y'],
                                         cfg['omega_c'], cfg['W_cut'], cfg['alpha'])
            loss_kl = q.kl_divergence(it, cfg['iters']) / q.dim
            loss_cum = cumulant_regularizer(q.mu) if cfg['lambda_cum'] > 0 else torch.tensor(0.0, device=self.device)
            loss = loss_dc + cfg['beta_kl'] * loss_kl + cfg['lambda_cum'] * loss_cum

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([q.mu], max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                print(f'  warning: non-finite loss at iteration {it}')
                optimizer.zero_grad()
            del z0, loss_dc, loss_kl, loss, x1

        return q.mu.detach().clone()

    # ---- Stage 2 -----------------------------------------------------------
    def data_consistency(self, z0, A, y, sigma, step_scale, sigma_y, dc_steps, eta_tilde=0.8, phi=1.0):
        """Measurement-consistency correction of the clean latent: a few gradient steps on a
        back-projection / least-squares mixture, run through the decoder."""
        lam = ((1 - sigma).clamp(0, 1)) ** phi
        w_bp, w_ls = (1 - lam), lam
        eta = max(1e-4, sigma_y ** 2 * eta_tilde)

        z = z0.detach().to(self.device).requires_grad_(True)
        step = (step_scale * (0.25 + 0.75 * float(sigma) ** 2)) / dc_steps
        y_flat = y.reshape(y.size(0), -1).contiguous()
        for _ in range(dc_steps):
            x = self.decode(z.to(dtype=self.dtype)).float()
            Ax = A.A(x.reshape(x.size(0), -1).contiguous())
            loss = (w_bp * torch.linalg.norm(A.A_pinv_add_eta(Ax, eta) - A.A_pinv_add_eta(y_flat, eta))
                    + w_ls * torch.linalg.norm(Ax - y_flat))
            grad = torch.autograd.grad(loss, z)[0].to(dtype=self.dtype)
            z = (z - step * grad).detach().requires_grad_(True)
        return z.detach().to(dtype=z0.dtype)

    def stage2(self, x1, y, A, emb, null, cfg):
        """Posterior sampling with bounded stochasticity from the optimized initial noise x1."""
        steps = cfg['stage2_steps']
        emb = [e.to(self.device, dtype=torch.float16) for e in emb]
        null = [e.to(self.device, dtype=torch.float16) for e in null]
        z = x1.to(dtype=torch.float16)
        timesteps, sigmas = self.schedule(steps)

        for i in range(steps):
            u = i / (steps - 1)
            step_scale = cfg['beta_start'] + (cfg['beta_end'] - cfg['beta_start']) * u
            t = timesteps[i].expand(z.shape[0]).to(self.device)
            with torch.no_grad(), _autocast(torch.float16):
                v = self.velocity(z, t, emb, null, cfg['cfg_scale']).to(torch.float16)

            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            x0 = z - sigma.to(torch.float16) * v              # Tweedie estimate of the clean latent
            x1_hat = z + (1 - sigma).to(torch.float16) * v     # estimate of the source noise
            x0 = self.data_consistency(x0, A, y, sigma, step_scale, cfg['sigma_y'], cfg['dc_steps'])

            # Re-noise to sigma_next; the injected noise is eta(t) = lambda_s * sqrt(1 - sigma) * sigma.
            eps = torch.randn_like(x1_hat, dtype=torch.float16)
            a = min(float((1 - sigma_next) ** 0.5) * cfg['lambda_s'], 1.0)
            x1_mix = ((1 - a ** 2) ** 0.5) * x1_hat + a * eps
            z = (1 - sigma_next) * x0.to(torch.float16) + sigma_next * x1_mix

        with torch.no_grad(), _autocast(torch.float16):
            img = self.decode(z)
        return (img.float() / 2 + 0.5).clamp(0, 1).detach().cpu()
