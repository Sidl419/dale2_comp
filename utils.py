import cv2
import numpy as np


def PSNR(original, compressed):
    return 10 * np.log10(1 / np.mean((original - compressed) ** 2))


def load_image(path="images/1.png", size=256):
    image = cv2.imread(path)[..., ::-1]
    max_size = max(image.shape[:2])

    res_image = np.zeros((max_size, max_size, 3))
    res_image[: image.shape[0], : image.shape[1]] = image

    res_image = cv2.resize(res_image, (size, size))
    res_image = res_image.astype(np.float64) / 255.0
    return res_image
