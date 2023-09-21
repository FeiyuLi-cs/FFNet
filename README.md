# FFNet
Joint Demosaicing and Denoising with Frequency Domain Features

## Contents
1. [Environment](#env)
2. [Train](#train)
3. [Test](#test)
4. [Dataset](#data)
5. [Other](#other)

## Environment <a name="env"></a>
```shell
python=3.8 numpy=1.21.2 opencv-python=4.5.5.64
pillow=8.4.0 numba=0.55.1 scikit-image=0.18.3
pytorch=1.10.0 torchvision=0.11.1 cudatoolkit=11.3
```

## Train <a name="train"></a>
train JDD:
```shell
python train.py --phase train --task JDnDm --model FFNet --in_type noisy_rgb
```

train DM:
```shell
python train.py --phase train --task DM --model FFNet-DM-B --in_type rgb
```
If you want to train other model, please change ```--model "your model name"```. The model weights will be saved in ```./logs/.../checkpoint/xxx.pth``` folder.

## Test <a name="test"></a>
Our pretrain models in [Google drive](https://drive.google.com/drive/folders/1RaXeZnENpQdzxftBHzm0ffLvAZf8PkW4?usp=sharing).

For JDD:

To test FFNet, run the command below:
```shell
python jdd_test.py --phase test --model FFNet --test_path dataset/JDnDm/test/ --pretrain logs/JDnDm/DIV2K/model_FFNet-in_type_noisy_rgb-C_64-B_32-Patch_128-Epoch_200/checkpoint/model_FFNet-in_type_noisy_rgb-C_64-B_32-Patch_128-Epoch_200_checkpoint_best.pth
```

For DM:

To test FFNet-DM-B, run the command below:
```shell
python dm_test.py --phase test --model FFNet-DM-B --test_path dataset/JDnDm/test/ --pretrain logs/JDnDm/DIV2K/model_FFNet-DM-B-in_type_rgb-C_64-B_32-Patch_128-Epoch_200/checkpoint/model_FFNet-DM-B-in_type_rgb-C_64-B_32-Patch_128-Epoch_200_checkpoint_best.pth 
```

The test logs will be saved in ```./logs/.../xxx_test.log``` folder and results will be saved in ```./logs/.../results/...``` folder.

## Dataset <a name="data"></a>
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) train dataset and [Kodak](http://r0k.us/graphics/kodak/), [McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm) and [Urban100](https://github.com/jbhuang0604/SelfExSR) test datasets (you can also download test datasets in our [Google Drive](https://drive.google.com/drive/folders/1RaXeZnENpQdzxftBHzm0ffLvAZf8PkW4?usp=sharing)).

You can obtain the train subdataset from [BasicSR](https://github.com/XPixelGroup/BasicSR) by `scipts/data_preparation/extract_subimages.py` (only extract HR images).

The data folder should be like the format below:
```
dataset
├─ DIV2K
│ ├─ train     % 32592 images
│ │ ├─ DIV2K_train_HR_sub
│ │ |  ├─ xxxx.png
│ │ |  ├─ ...
│ | |
| | |
│ ├─ valid     % 4152 images
│ │ ├─ DIV2K_valid_HR_sub
│ │ |  ├─ xxxx.png
│ │ |  ├─ ...
|
|
├─ JDnDM
│ ├─ test
| │ ├─ Kodak     
| │ │ ├─ xxxx.png
│ | | ├─ ......
│ │ |
| | |
│ | ├─ McMaster
│ │ | ├─ xxxx.png
│ │ | ├─ ......
| | |
| | |
│ | ├─ Urban100   
│ │ | ├─ xxxx.png
│ │ | ├─ ......
...
```

## Other <a name="other"></a>
For JDD: [JDNDMSR](https://github.com/xingwz/End-to-End-JDNDMSR), [CDLNet](https://github.com/nikopj/CDLNet-OJSP).

For DM: [IRCNN](https://github.com/cszn/DPIR), [DPIR](https://github.com/cszn/DPIR), [RSTCANet](https://github.com/xingwz/RSTCANet).