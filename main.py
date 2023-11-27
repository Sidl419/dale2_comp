import torch
import glob
from dalle2_pytorch.train_configs import TrainDecoderConfig
from utils import load_image, PSNR
from diffusion import generate_diffusion
from optimize import optimize
from matplotlib import pyplot as plt


def main():
    decoder_config = TrainDecoderConfig.from_json_path("models/second_decoder_config.json").decoder
    decoder_config.sample_timesteps = 100
    decoder_config.ddim_sampling_eta = 0
    latent_size = 64
    conditional_size = 64
    start_at_unet_number = 1
    decoder = decoder_config.create()
    decoder_model_state = torch.load("models/second_decoder.pth", map_location=torch.device("cuda"))
    decoder.load_state_dict(decoder_model_state, strict=False)

    print("Decoder is ready")
    decoder = decoder.cuda()

    for image_path in glob.glob("images/*"):
        base_image = [load_image(path=image_path, size=256)]
        scaled_image = [load_image(path=image_path, size=conditional_size)]
        print(f"Loaded image {image_path}")
        image_name = image_path.split("/")[-1].split(".")[0]

        cond_image = torch.tensor(scaled_image, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        target_image = torch.tensor(base_image, dtype=torch.float32).permute(0, 3, 1, 2).cuda()

        with open(f"texts/{image_name}.txt", "r") as file:
            text_prompt = [file.read()]
        print(f"Loaded text: \n{text_prompt[0]}")

        latent = generate_diffusion(
            target_image,
            decoder,
            cond_image,
            text_prompt,
            start_at_unet_number,
            mode="backward",
        )
        latent = torch.nn.functional.interpolate(latent, size=(latent_size, latent_size), mode="area")
        init_image = torch.nn.functional.interpolate(latent, size=(256, 256), mode="area")

        learn_cond_image = optimize(
            decoder, text_prompt, target_image, cond_image, start_at_unet_number
        )
        pred_image = generate_diffusion(
            init_image,
            decoder,
            learn_cond_image,
            text_prompt,
            start_at_unet_number,
            mode="forward",
        )
        pred_image = pred_image.permute(0, 2, 3, 1).cpu().numpy()[0]
        latent = latent.permute(0, 2, 3, 1).cpu().numpy()[0]
        learn_cond_image = learn_cond_image.permute(0, 2, 3, 1).cpu().numpy()[0]
        curr_psnr = PSNR(pred_image, base_image[0])

        plt.suptitle(f"RGB-PSNR is {round(curr_psnr, 3)}")
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(base_image[0])
        plt.subplot(2, 2, 2)
        plt.title("Latent Image")
        plt.imshow(latent)
        plt.subplot(2, 2, 3)
        plt.title("Learned Conditional Image")
        plt.imshow(learn_cond_image)
        plt.subplot(2, 2, 4)
        plt.title("Result Image")
        plt.imshow(pred_image)
        plt.tight_layout()
        plt.savefig(f"result/{image_name}.png")
        print(f"Result saved to result/{image_name}.png")

        print(f"RGB-PSNR is {curr_psnr}")


if __name__ == "__main__":
    main()
