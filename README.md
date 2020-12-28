# TDN: Temporal Difference Networks for Efficient Action Recognition

![1](https://github.com/MCG-NJU/TDN/blob/main/TDM.jpg)  
## Overview

We release the PyTorch code of the [TDN](https://arxiv.org/abs/2012.10071)(Temporal Difference Networks). This code is based on the [TSN](https://github.com/yjxiong/tsn-pytorch) and [TSM](https://github.com/mit-han-lab/temporal-shift-module) codebase. The core code to implement the Temporal Difference Module are `ops/base_module.py` and `ops/tdn_net.py`.


**🔥 [NEW!]**  We have released the PyTorch code of TDN. 
* [Prerequisites](#prerequisites)
* [Data Preparation](#data-preparation)
* [Model Zoo](#model-zoo)
* [Testing](#testing)  
* [Training](#training)  

## Prerequisites
The code is built with following libraries:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) **1.4** or higher
- [Torchvision](https://github.com/pytorch/vision)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)
- [ffmpeg](https://www.ffmpeg.org/)  
- [decord](https://github.com/dmlc/decord) 

## Data Preparation
We have successfully trained TDN on [Kinetics400](https://deepmind.com/research/open-source/kinetics), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2) with this codebase.  
- The processing of Something-Something-V1 & V2 can be summarized into 3 steps:  
    1. Extract frames from videos(you can use ffmpeg to get frames from video)      
    2. Generate annotations needed for dataloader ("<path_to_frames> <frames_num> <video_class>" in annotations) The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
        ```
        frames/video_1 num_frames label_1
        frames/video_2 num_frames label_2
        frames/video_3 num_frames label_3
        ...
        frames/video_N num_frames label_N
        ```
    3. Add the information to ops/dataset_configs.py  

- The processing of Kinetics400 can be summarized into 2 steps:  
    1. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations) The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
        ```
        frames/video_1.mp4  label_1
        frames/video_2.mp4  label_2
        frames/video_3.mp4  label_3
        ...
        frames/video_N.mp4  label_N
        ```
    2. Add the information to ops/dataset_configs.py  

## Model Zoo
Here we provide some off-the-shelf [pretrained models](https://drive.google.com/drive/folders/1du1KtjFANeD0Mr5_7RxCx0jLUVRj_y8w?usp=sharing). The accuracy might vary a little bit compared to the [paper]((https://arxiv.org/abs/2012.10071)), since the raw video of Kinetics downloaded by users may have some differences. 
#### Something-Something-V1

Model  | Frames x Crops x Clips  | Top-1  | Top-5 | checkpoint
:--: | :--: | :--: | :--:| :--:
TDN-ResNet50  | 8x1x1 | 52.3%  | 80.6% | [link](https://drive.google.com/file/d/13PeSkxDgYZ_Z-OOYiuCY3J_QlY2noUuv/view?usp=sharing)
TDN-ResNet50  | 16x1x1 | 53.9%  | 82.1% | [link](https://drive.google.com/file/d/1IxUMglMEET8blZxzJrq7sc79MyosJ52I/view?usp=sharing)

#### Something-Something-V2

Model  | Frames x Crops x Clips | Top-1  | Top-5 | checkpoint
:--: | :--: | :--: | :--:| :--:
TDN-ResNet50  | 8x1x1   | 64.0%   | 88.8%  | [link](https://drive.google.com/file/d/1cPgY2h6zVdjKrFJu-ZV_uo_8HKLPh_Vu/view?usp=sharing)
TDN-ResNet50  | 16x1x1  | 65.3%   | 89.7%  | [link](https://drive.google.com/file/d/1pwkrPYUc36b5SnPRHUbOcycm4zoQHZnR/view?usp=sharing)

#### Kinetics400
Model  | Frames x Crops x Clips   | Top-1 (30 view)  | Top-5 (30 view)  | checkpoint
:--: | :--: | :--: | :--:| :--:
TDN-ResNet50    | 8x3x10  | 76.6%  | 92.8%  | [link](https://drive.google.com/file/d/1h-yoGaQYpRoSb57-sw_Iw7mYkSrI5QnQ/view?usp=sharing)
TDN-ResNet50    | 16x3x10 | 77.5%  | 93.2%  | [link](https://drive.google.com/file/d/1M6nkUISaIfLAxuabA3_JxPUiXqnLv26T/view?usp=sharing)
TDN-ResNet101   | 8x3x10 | 77.5%  | 93.6%  | [link](https://drive.google.com/file/d/1UdvGQkBkC9vqOwknXbyuuTVDTp1ctOr4/view?usp=sharing)
TDN-ResNet101   | 16x3x10 | 78.5%  | 93.9%  | [link](https://drive.google.com/file/d/1MWj3srcdJginXCZfE6P4ufPRpLbUZe-Z/view?usp=sharing)

## Testing
- For center crop single clip, the processing of testing can be summarized into 2 steps:
    1. Run the following testing scripts:
        ```
        CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
        --test_crops=1 --batch_size=16 --output_dir <your_pkl_path> -j 4 --clip_index=1
        ```
    2. Run the following scripts to get result from the raw score:
        ```
        python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>  
        ```
- For 3 crops, 10 clips, the processing of testing can be summarized into 2 steps: 
    1. Run the following testing scripts for 10 times(clip_index from 0 to 9):
        ``` 
        CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
        --test_crops=3 --batch_size=16 --full_res --gpus --output_dir <your_pkl_path>  \
        -j 4 --clip_index <your_clip_index>
        ```
    2. Run the following scripts to ensemble the raw score of the 30 views:
        ```
        python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir <your_pkl_path> 
        ```
## Training
This implementation supports multi-gpu, `DistributedDataParallel` training, which is faster and simpler. 
- For example, to train TDN-ResNet50 on Something-Something-V1 with 8 gpus, you can run:
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
                main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
                --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 16 \
                --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```
- For example, to train TDN-ResNet50 on Kinetics400 with 8 gpus, you can run:
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
            main.py  kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
            --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 16 \
            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```
## Acknowledgements
We especially thank the contributors of the [TSN](https://github.com/yjxiong/tsn-pytorch) and [TSM](https://github.com/mit-han-lab/temporal-shift-module) codebase for providing helpful code.
## License
This repository is released under the Apache-2.0. license as found in the [LICENSE](https://github.com/MCG-NJU/TDN/blob/main/LICENSE) file.
## Citation
If you think our work is useful, please feel free to cite our paper 😆 :
```
@article{wang2020tdn,
      title={TDN: Temporal Difference Networks for Efficient Action Recognition}, 
      author={Limin Wang and Zhan Tong and Bin Ji and Gangshan Wu},
      journal={arXiv preprint arXiv:2012.10071},
      year={2020}
}
```



