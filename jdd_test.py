import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from config.jdd_config import BaseArgs
from dataload.data_util import mosaic, add_noise_test
from utils.metrics import calculate_psnr, calculate_ssim
from utils.model_util import load_pretrained_models
from model import cal_model_parm_nums, build_net


def main(args):
    print('\n==========     Loading the network     =============')
    model = build_net(args)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f Mb' % (model_size / 1e6))

    # load pre-trained
    pretrain = args.pretrain
    model, _, _ = load_pretrained_models(model, pretrain)

    test_target_path = args.test_path
    test_output_path = args.save_dir
    # print(test_output_path)
    noiseSet = [5, 10, 15, 20, 25, 30]
    with torch.no_grad():

        for filename1 in os.listdir(test_target_path):
            test_target_path_child = test_target_path + filename1 + '/'
            test_output_path_child = test_output_path + filename1 + '/'
            if not os.path.isdir(test_output_path_child):
                os.makedirs(test_output_path_child)

            log = open(os.path.join(args.exp_dir, '{}_{}.log'.format(filename1, args.phase)), 'w')

            for noise in noiseSet:

                for filename2 in os.listdir(test_target_path_child):
                    image = test_target_path_child + filename2
                    if not os.path.isdir(test_output_path_child + str(noise) + '/'):
                        os.makedirs(test_output_path_child + str(noise) + '/')

                    raw_image_tensor = ImageToTensor(image)
                    raw_image_tensor = add_noise_test(raw_image_tensor, noise)
                    raw_image_tensor = raw_image_tensor.unsqueeze(0)
                    raw_image_tensor = raw_image_tensor.to(device)
                    output = model(raw_image_tensor)
                    save_image(output, '%s%s.png' %
                               (test_output_path_child + str(noise) + '/', filename2.split('.')[0]))

                print('\n==========     noisy:{} PSNR SSIM     ============='.format(noise))
                log.write('==========     noisy:%s     =============\n' % json.dumps(noise))
                # print(test_target_path_child, resizeDimension)
                psnr_ssim(test_target_path_child, test_output_path_child + str(noise) + '/', log)
        log.close()
        print('Number of params: %.4f Mb' % (model_size / 1e6))


def ImageToTensor(image):
    image = Image.open(image)

    test_transform = transforms.Compose(
        [
            # transforms.Resize(resizeDimension),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ]
    )
    gt_tensor = test_transform(image)
    raw_image_tensor = mosaic(gt_tensor)
    return raw_image_tensor


def psnr_ssim(test_target_path, test_output_path, log):
    state = {
        'IMG': '',
        'time': '',
        'psnr': 0.0,
        'ssim': 0.0,
    }

    avg_psnr = 0
    avg_ssim = 0
    count = 0

    for filename in os.listdir(test_target_path):
        img1_path = test_target_path + filename
        img2_path = test_output_path + filename

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)

        print('{} \tPSNR: {} \tSSIM: {} '.format(filename, psnr, ssim))
        state['IMG'] = filename
        state['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state['psnr'] = psnr
        state['ssim'] = ssim
        avg_psnr += psnr
        avg_ssim += ssim
        count += 1
        log.write('%s\n' % json.dumps(state))
        log.flush()

    print('total img count ', count)
    print('PSNR {} SSIM {}'.format(avg_psnr / count, avg_ssim / count))
    log.write('%s\n\n' % json.dumps(
        {'Total Count': count,
         'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
         'avg_psnr': avg_psnr / count,
         'avg_ssim': avg_ssim / count}))
    log.flush()
    print('done!!!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    main(args)
