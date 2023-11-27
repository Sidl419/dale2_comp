import torch
from tqdm import tqdm
from dalle2_pytorch.dalle2_pytorch import resize_image_to, maybe
from dalle2_pytorch.tokenizer import tokenizer


def diffusion_step(image, decoder, unet, alpha, alpha_next, time, text_encodings, lowres_cond_img):
    time_cond = torch.full((image.shape[0],), time, device=decoder.device, dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred = unet.forward(
            image,
            time_cond,
            image_embed=lowres_cond_img,
            text_encodings=text_encodings,
            lowres_cond_img=lowres_cond_img,
        )

    sigma = (1 - alpha).sqrt()
    sigma_next = (1 - alpha_next).sqrt()
    x_0 = (image - sigma * pred) / alpha.sqrt()
    x_0 = decoder.dynamic_threshold(x_0)
    pred_noise = (image - x_0 * alpha.sqrt()) / sigma

    return x_0 * alpha_next.sqrt() + pred_noise * sigma_next


@torch.no_grad()
def diffusion_pass(
    latent,
    decoder,
    unet,
    text_encodings,
    cond_img,
    alphas,
    time_pairs
):
    image = latent.detach().clone()
    cond_img = maybe(decoder.normalize_img)(cond_img)

    for time, time_next in tqdm(time_pairs):
        image = diffusion_step(
            image, decoder, unet, alphas[time], alphas[time_next], time, text_encodings, cond_img
        )

    return image


def generate_diffusion(latent, decoder, image, text, start_at_unet_number=1, mode="backward"):
    _, text_encodings = decoder.clip.embed_text(tokenizer.tokenize(text).cuda())
    image = resize_image_to(image, decoder.image_sizes[start_at_unet_number-1], nearest=True)

    for (
        unet, vae, image_size, noise_scheduler, sample_timesteps
    ) in zip(
        decoder.unets[start_at_unet_number:],
        decoder.vaes[start_at_unet_number:],
        decoder.image_sizes[start_at_unet_number:],
        decoder.noise_schedulers[start_at_unet_number:],
        decoder.sample_timesteps[start_at_unet_number:],
    ):

        lowres_cond_img = resize_image_to(
            image,
            target_image_size=image_size,
            clamp_range=decoder.input_image_range,
            nearest=True,
        )
        lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        if mode == "backward":
            times = torch.linspace(0, noise_scheduler.num_timesteps, sample_timesteps + 2)[:-1].int().tolist()
            time_pairs = list(zip(times[1:], times[:-1]))
        else:
            times = torch.linspace(noise_scheduler.num_timesteps, 0, sample_timesteps + 2)[1:].int().tolist()
            time_pairs = list(zip(times[:-1], times[1:]))

        latent = diffusion_pass(
            latent=latent,
            decoder=decoder,
            unet=unet,
            text_encodings=text_encodings,
            cond_img=lowres_cond_img,
            alphas=noise_scheduler.alphas_cumprod,
            time_pairs=time_pairs,
        )
        if mode == "forward":
            latent = decoder.unnormalize_img(latent)

    return latent
