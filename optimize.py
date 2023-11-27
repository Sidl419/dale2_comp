import torch
from tqdm import tqdm
from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.dalle2_pytorch import resize_image_to, maybe
from diffusion import diffusion_step
from utils import PSNR

def optimize(decoder, text, target_image, cond_image, start_at_unet_number=1, epoch_num=100):
    _, text_encodings = decoder.clip.embed_text(tokenizer.tokenize(text).cuda())
    best_image = None

    for (
        unet, vae, image_size, noise_scheduler, sample_timesteps
    ) in zip(
        decoder.unets[start_at_unet_number:],
        decoder.vaes[start_at_unet_number:],
        decoder.image_sizes[start_at_unet_number:],
        decoder.noise_schedulers[start_at_unet_number:],
        decoder.sample_timesteps[start_at_unet_number:],
    ):
        learn_image = torch.tensor(cond_image.data, requires_grad=True)
        optimizer = torch.optim.SGD(params=[learn_image], lr=2)

        image_size_encoded = vae.get_encoded_fmap_size(image_size)
        target_image_normalized = maybe(decoder.normalize_img)(
            resize_image_to(
                target_image,
                target_image_size=image_size_encoded,
                clamp_range=decoder.input_image_range,
                nearest=True,
            )
        )

        times = torch.linspace(
            noise_scheduler.num_timesteps, 0, sample_timesteps + 2
        )[1:].int().tolist()
        time_pairs = list(zip(times[:-1], times[1:]))

        best_psnr = 0.0
        best_image = None

        for epoch in tqdm(range(epoch_num)):
            image = torch.randn_like(target_image)

            for time, time_next in time_pairs:
                optimizer.zero_grad()

                lowres_cond_img = maybe(decoder.normalize_img)(
                    resize_image_to(
                        learn_image,
                        target_image_size=image_size_encoded,
                        clamp_range=decoder.input_image_range,
                        nearest=True,
                    )
                )

                prediction = diffusion_step(
                    image,
                    decoder,
                    unet,
                    alpha=noise_scheduler.alphas_cumprod[time],
                    alpha_next=noise_scheduler.alphas_cumprod[time_next],
                    time=time,
                    text_encodings=text_encodings,
                    cond_img=lowres_cond_img,
                )

                loss = torch.nn.functional.mse_loss(
                    prediction, target_image_normalized
                )
                loss.backward()
                optimizer.step()

                image = prediction.detach()

            psnr = PSNR(
                decoder.unnormalize_img(image)[0].permute(1, 2, 0).cpu().numpy(),
                target_image[0].permute(1, 2, 0).cpu().numpy(),
            )

            if psnr > best_psnr:
                best_psnr = psnr
                best_image = learn_image.clone()

            print(f"Epoch: {epoch} loss: {loss.item():.3f}, psnr: {psnr:.3f}")
  
    if best_image:
        print(f"Best PSNR: {best_psnr:.3f}")
        return best_image
    else:
        return cond_image
