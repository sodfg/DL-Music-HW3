"""Adaptive Projected Guidance (APG) for Stable Audio Open.

Sadat et al., "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models", ICLR 2025.
CFG adds (w-1) * (D_cond - D_uncond). APG splits that update into a part parallel to D_cond (which mostly
raises amplitude / saturation) and an orthogonal part (which carries the content change), and down-weights the
parallel part by eta. A negative-momentum running average and an optional norm cap further damp the update.

APG is defined on the denoised prediction D (x0 estimate), not on v. Stable Audio's scheduler is EDM-style:
    D = c_skip * x + c_out * v,  c_skip = sd^2 / (s^2 + sd^2),  c_out = -s * sd / sqrt(s^2 + sd^2)
so each step we map both v outputs to D, apply APG there, and map the result back to v.
"""

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed


class MomentumBuffer:
    def __init__(self, momentum):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value):
        self.running_average = update_value + self.momentum * self.running_average


def project(v0, v1):
    """Split v0 into components parallel and orthogonal to v1 (per sample, over all non-batch dims)."""
    dims = list(range(1, v0.ndim))
    v0, v1 = v0.double(), v1.double()
    v1 = v1 / v1.norm(dim=dims, keepdim=True)
    parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    return parallel.float(), (v0 - parallel).float()


def apg(den_uncond, den_cond, guidance_scale, eta=0.0, momentum_buffer=None, norm_threshold=0.0):
    """APG on denoised predictions. eta=1, no momentum, no norm cap reduces exactly to CFG."""
    diff = den_cond - den_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        dims = list(range(1, diff.ndim))
        diff_norm = diff.norm(dim=dims, keepdim=True)
        diff = diff * torch.clamp(norm_threshold / diff_norm, max=1.0)
    parallel, orthogonal = project(diff, den_cond)
    return den_cond + (guidance_scale - 1) * (orthogonal + eta * parallel)


@torch.no_grad()
def generate_apg(pipe, prompt, guidance_scale=7.0, num_inference_steps=100, audio_end_in_s=10.0, seed=0,
                 eta=0.0, momentum=-0.5, norm_threshold=0.0):
    """Same sampling loop as sample_code/generate.py, with the CFG step replaced by APG in D-space."""
    device = pipe._execution_device
    sampling_rate = pipe.vae.config.sampling_rate
    sigma_data = pipe.scheduler.config.sigma_data

    # 1. Conditions: [uncond, cond]. The unconditional branch is an all-zero text+duration embedding.
    prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
    seconds_start, seconds_end = pipe.encode_duration(0.0, audio_end_in_s, device, False, batch_size=1)
    cond = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
    encoder_hidden_states = torch.cat([torch.zeros_like(cond), cond])
    global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)

    # 2. Timesteps, initial noise, rotary embedding
    generator = torch.Generator(device).manual_seed(seed)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
        cond.dtype, device, generator, audio_channels=pipe.vae.config.audio_channels,
    )
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.0)
    rotary_embedding = get_1d_rotary_pos_embed(
        pipe.rotary_embed_dim, latents.shape[2] + global_hidden_states.shape[1],
        use_real=True, repeat_interleave_real=False,
    )
    momentum_buffer = MomentumBuffer(momentum) if momentum != 0 else None

    # 3. Denoising loop
    for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
        latent_model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = pipe.transformer(
            latent_model_input, t.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            global_hidden_states=global_hidden_states,
            rotary_embedding=rotary_embedding,
            return_dict=False,
        )[0]
        v_uncond, v_cond = noise_pred.float().chunk(2)

        sigma = pipe.scheduler.sigmas[i].to(device=device, dtype=torch.float32)
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        x = latents.float()
        den_guided = apg(c_skip * x + c_out * v_uncond, c_skip * x + c_out * v_cond, guidance_scale,
                         eta=eta, momentum_buffer=momentum_buffer, norm_threshold=norm_threshold)
        noise_pred = ((den_guided - c_skip * x) / c_out).to(latents.dtype)

        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    # 4. Decode
    audio = pipe.vae.decode(latents).sample
    return audio[0, :, : int(audio_end_in_s * sampling_rate)]
