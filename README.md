# Dalle2 Semantic Compression

Semantic compression needs to be implemented, which, based on a textual description of an image, reconstructs the image itself. Semantic compression translates the textual description into text embedding; then optionally, due to the modality gap, using a prior, it translates the text embedding into image embedding; then, with the help of the decoder, obtains a low-resolution image, for example, 64x64, from the image embedding; then, with the help of the upsampler, enhances the image quality, for example, to 256x256 or 512x512.

Average RGB-PSNR with [that decoder](https://huggingface.co/Veldrovive/upsamplers/resolve/main/working/latest.pth) and
[its config](https://huggingface.co/Veldrovive/upsamplers/raw/main/working/decoder_config.json) is 40.

## Installation

```bash
git clone https://github.com/Sidl419/dale2_comp.git
poetry install # we use poetry for versioning
```

## Usage

Put the decoder configuration and weights in `models` folder, input images should be put to `images` and input texts to
`texts`. The result of the model would be kept in `result` folder. Then run

```bash
python main.py
```

## Main workflow

The `main.py` works as follows:

* **Decoder Setup:**
    Reads the decoder configuration from models/second_decoder_config.json.
    Sets configuration parameters such as sample timesteps, latent size, and conditional size.
    Loads the pre-trained decoder model from models/second_decoder.pth.

* **Data Loading:**
    Loads input images, scales them to the required sizes, and prints information about the loaded images.

* **Diffusion Process:**
    Utilizes the diffusion model to optimize images based on provided text prompts.
    Saves the resulting images, latent images, and learned conditional images.

* **Visualization:**
    Calculates and displays the RGB-PSNR (Peak Signal-to-Noise Ratio) for each processed image.
    Plots a visual comparison of the original image, latent image, learned conditional image, and result image.
