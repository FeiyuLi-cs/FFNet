import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from dataload.data_util import mosaic, add_noise_train, add_noise_test


class LoadJDDDIV2KData(Dataset):
    def __init__(self, image_path, mode, patch_size, in_type, min_noise, max_noise):
        self.mode = mode
        self.patch_size = patch_size
        self.in_type = in_type
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.image_list = imageList(image_path)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]
        )
        self.valid_transform = transforms.Compose(
            [
                transforms.CenterCrop(patch_size),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # Read Images
        gt_image = Image.open(self.image_list[i])
        if self.mode == 'train':
            gt_image_tensor = self.train_transform(gt_image)
        else:
            gt_image_tensor = self.valid_transform(gt_image)

        raw_image_tensor = mosaic(gt_image_tensor)
        if 'noisy' in self.in_type:
            if self.mode == 'train':
                raw_image_tensor = add_noise_train(raw_image_tensor, self.min_noise, self.max_noise)
            else:
                raw_image_tensor = add_noise_test(raw_image_tensor, sigma=10)

        return raw_image_tensor, gt_image_tensor


def imageList(path, multiDir=False, imageExtension=None):
    if imageExtension is None:
        imageExtension = ['*.jpg', '*.png', '*.jpeg', '*.tif', '*.bmp']
    imageList = []
    for ext in imageExtension:
        if multiDir == True:
            imageList.extend(glob.glob(path + "*/" + ext))
        else:
            imageList.extend(glob.glob(path + ext))
    return imageList


if __name__ == '__main__':
    print('noisy' in 'noisy_raw')
    print('lin' in 'noisy_raw_lin')
    # JDnDmSR
