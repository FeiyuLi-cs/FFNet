import random
import numpy as np

import torch


def mosaic(image_tensor):
    """Extracts RGGB Bayer planes from an RGB image."""
    red = image_tensor[0, 0::2, 0::2]
    green_red = image_tensor[1, 0::2, 1::2]
    green_blue = image_tensor[1, 1::2, 0::2]
    blue = image_tensor[2, 1::2, 1::2]
    out = torch.stack((red, green_red, green_blue, blue), dim=0)
    return out


def add_noise_train(img_tensor, min_noise, max_noise):
    sigma = max(min_noise, np.random.rand(1) * max_noise)[0]
    noise = torch.randn(img_tensor.size()).mul_(sigma)
    img_tensor = img_tensor + noise
    return img_tensor


def add_noise_test(img_tensor, sigma):
    random.seed(0)
    torch.manual_seed(0)
    noise = torch.randn(img_tensor.size()).mul_(sigma / 255.)
    img_tensor = img_tensor + noise
    return img_tensor
