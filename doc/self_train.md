## somthing

### TDN_resnet50_lr(0.02)_epoch(100)-top1=49.64(8x1x1)
2021.0629
`just like below, one epoch cost 1.52h when we trained in 2*nvidia2080ti batch(8); 100 epoch cost 3day 6h.`

`python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
        main.py  something RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
        --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 8 \
        --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --decoded_type decord`

- 8x1x1

something: 174 classes

​    Initializing TSN with base model: resnet50.

​    TSN Configurations:

​        input_modality:     RGB

​        num_segments:       8

​        new_length:         1

​        consensus_module:   avg

​        dropout_ratio:      0.8

​        img_feature_dim:    256   

=> base model: resnet50

video number:11522

video 0 done, total 0/11522, average 0.13565 sec/video, moving Prec@1 53.125 Prec@5 87.500

video 640 done, total 640/11522, average 0.01348 sec/video, moving Prec@1 53.869 Prec@5 81.994

video 1280 done, total 1280/11522, average 0.01007 sec/video, moving Prec@1 52.515 Prec@5 79.878

video 1920 done, total 1920/11522, average 0.00890 sec/video, moving Prec@1 50.717 Prec@5 79.764

video 2560 done, total 2560/11522, average 0.00826 sec/video, moving Prec@1 50.772 Prec@5 79.745

video 3200 done, total 3200/11522, average 0.00791 sec/video, moving Prec@1 50.712 Prec@5 80.043

video 3840 done, total 3840/11522, average 0.00779 sec/video, moving Prec@1 50.181 Prec@5 79.029

video 4480 done, total 4480/11522, average 0.00767 sec/video, moving Prec@1 49.734 Prec@5 78.723

video 5120 done, total 5120/11522, average 0.00761 sec/video, moving Prec@1 49.398 Prec@5 78.339

video 5760 done, total 5760/11522, average 0.00747 sec/video, moving Prec@1 49.793 Prec@5 78.349

video 6400 done, total 6400/11522, average 0.00738 sec/video, moving Prec@1 49.642 Prec@5 78.467

video 7040 done, total 7040/11522, average 0.00735 sec/video, moving Prec@1 49.675 Prec@5 78.606

video 7680 done, total 7680/11522, average 0.00728 sec/video, moving Prec@1 49.637 Prec@5 78.410

video 8320 done, total 8320/11522, average 0.00722 sec/video, moving Prec@1 49.521 Prec@5 78.388

video 8960 done, total 8960/11522, average 0.00720 sec/video, moving Prec@1 49.322 Prec@5 78.392

video 9600 done, total 9600/11522, average 0.00714 sec/video, moving Prec@1 49.346 Prec@5 78.457

video 10240 done, total 10240/11522, average 0.00709 sec/video, moving Prec@1 49.426 Prec@5 78.631

video 10880 done, total 10880/11522, average 0.00705 sec/video, moving Prec@1 49.578 Prec@5 78.620

video 11520 done, total 11520/11522, average 0.00702 sec/video, moving Prec@1 49.644 Prec@5 78.676

100%|████████████████████████████████████████████████████████| 11522/11522 [00:00<00:00, 63475.77it/s]

-----Evaluation is finished------

Overall Prec@1 49.64% Prec@5 78.68%

but origin is :top1=52.3%; maybe our parameters is not good; and found lr and wd is not good; the origin paper use `lr=0.01,wd=5e-4,--lr_steps  30 45 55 --epochs 60`, our lr is double of paper, maybe this is reason.


- 8x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(60)-top1=49.64(8x1x1)
2021.0629
`just like below, one epoch cost 1.52h when we trained in 2*nvidia2080ti batch(8); 100 epoch cost 3day 6h.`

`python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \`
`            main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \`
 `           --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \`
 `           --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result`
`Overall Prec@1 52.23% Prec@5 80.32%; but origin is :top1=52.3%;`


## somthingv2
- 8x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=63.91(8x1x1)
2021.07.08
[TDN__somethingv2_RGB_resnet50_avg_segment8_e100](log/TDN__somethingv2_RGB_resnet50_avg_segment8_e100/TDN__somethingv2_RGB_resnet50_avg_segment8_e100.png)
`just like below, one epoch cost 1.52h when we trained in 2*nvidia2080ti batch(8); 100 epoch cost 1d20h 3m 32s.`

`python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \`
            `main.py  somethingv2  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \`
            `--lr_scheduler step --lr_steps  30 45 55 --epochs 100 --batch-size 8 \`
            `--wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result`
`Overall Prec@1 63.91% Prec@5 88.80%; but origin is top1=64.0%;`

- 16x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=62.42%(16x1x1)
2021.07.08
[TDN__somethingv2_RGB_resnet50_avg_segment8_e100](log/TDN__somethingv2_RGB_resnet50_avg_segment16_e100/log.txt)
`just like below, one epoch cost 1.52h when we trained in 2*nvidia2080ti batch(8); 100 epoch cost 1d20h 3m 32s.`

`python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \`
            `main.py  somethingv2  RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.01 \`
            `--lr_scheduler step --lr_steps  30 45 55 --epochs 100 --batch-size 8 \`
            `--wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result`
`Overall Prec@1 62.42% Prec@5 87.59%`


## kinetics400
- 8x3x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=75.23(8x3x1)
2021.07.08
[TDN__kinetics_frame_RGB_resnet50_avg_segment8_e100](log/TDN__kinetics_frame_RGB_resnet50_avg_segment8_e100/log.txt)
`07/08 10:17:25 to 07/11 04:28:06; cost 3day 6h;`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet50' --weights TDN__kinetics_frame_RGB_resnet50_avg_segment8_e100_best.pth.tar  --test_segments=8 \`
`--test_crops=3 --clip_index 3 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 75.23% Prec@5 92.08%%;` 
- 8x3x1(Frames*crops*clips)
`Best Prec@1: '73.95141196013289'`

- 3x3x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=70.65%(3x3x1)
2021.07.08
[TDN__kinetics_frame_RGB_resnet50_avg_segment8_e100](log/TDN__kinetics_frame_RGB_resnet50_avg_segment3_e100/log.txt)
`07/08 10:17:25 to 07/11 04:28:06; cost 3day 6h;`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet50' --weights  --test_segments=3 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 70.65% Prec@5 89.11%` 
- 8x1x1(Frames*crops*clips)
`Best Prec@1: '69.'`


- 1x3x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=57.31%(1x3x1)(pretrained from imagenet)
2021.07.20
[TDN__kinetics_frame_RGB_resnet50_avg_segment1_e100](log/TDN__kinetics_frame_RGB_resnet50_avg_segment1_e100/log.txt)
`07/19`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet50' --weights  --test_segments=1 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 57.31% Prec@5 78.62%` 
- drop `13.34` with [3x3x1](resnet50)


- 1x3x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=57.31%(1x3x1)(no pretrained from imagenet)
2021.07.20
[TDN__kinetics_frame_RGB_resnet50_avg_segment1_e100]()
`07/19`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet50' --weights  --test_segments=1 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 57.02% Prec@5 78.56%` 
- drop `0.29` with [1x3x1](resnet50)
`you will found pre-trained from imagenet incerase litter, it rise less than 1 point;`


- 1x3x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=45.75%(1x3x1)(no pretrained from imagenet)
2021.07.20
[TDN__kinetics_frame_RGB_resnet18_avg_segment1_e100](log/TDN__kinetics_frame_RGB_resnet18_avg_segment1_e100/log.txt)
`07/19`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 45.75% Prec@5 69.64%` 
- drop `11.56` with [1x3x1](resnet50)


- 3x3x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=61.19%(3x3x1)(no pretrained from imagenet)
2021.07.21
[TDN__kinetics_frame_RGB_resnet18_avg_segment3_e100](log/TDN__kinetics_frame_RGB_resnet18_avg_segment3_e100/log.txt)
`07/19`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet18' --weights  --test_segments=3 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 61.19% Prec@5 83.69%` 
-  `61.43` with [3x1x1](resnet18)
- rise `15.44` with [1x3x1](resnet18)


- 9x3x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=71.30%(9x3x1)(no pretrained from imagenet)
2021.07.26
[TDN__kinetics_frame_RGB_resnet18_avg_segment9_e100](log/TDN__kinetics_frame_RGB_resnet18_avg_segment9_e100_True/log.txt)
`07/26`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet18' --weights  --test_segments=9 \`
`--test_crops=3 --clip_index 0 --batch_size=16 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 71.30% Prec@5 90.11%` 
-  `70` with [9x1x1](resnet18)
- rise `10.11` with [3x3x1](resnet18)


- 5x3x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=67.86%(5x3x1)(no pretrained from imagenet)
2021.07.28
[log/TDN__kinetics_frame_RGB_resnet18_avg_segment5_e100](log/TDN__kinetics_frame_RGB_resnet18_avg_segment5_e100_True/log.txt)
`07/28`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  kinetics_frame \`
`--archs='resnet18' --weights  --test_segments=5 \`
`--test_crops=3 --clip_index 0 --batch_size=16 --full_res --output_dir ./result  \`
`-j 4`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result`


`Overall Prec@1 67.86% Prec@5 87.73%` 
-  `67.27083` with [5x1x1](resnet18)
- drop `3.44` with [9x3x1](resnet18)



## high-events:train_A(2 classes)
`2 label`


- 1x3x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=98.55%(1x3x1)(no pretrained from imagenet)
2021.07.27
[TDN__high_events_RGB_resnet18_avg_segment1_e100](log/TDN__high_events_RGB_resnet18_avg_segment1_e100_True/log.txt)
`07/27`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_three_crops.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=3 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result --topk 2`


`Overall Prec@1 98.55% Prec@5 100.00%` 


- 1x1x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=100.00%(1x1x1)(no pretrained from imagenet)
2021.07.27
[TDN__high_events_RGB_resnet18_avg_segment1_e100](log/TDN__high_events_RGB_resnet18_avg_segment1_e100_True/log.txt)
`07/27`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 2`


`Overall Prec@1 100.00% Prec@5 100.00%` 

## high-events:train_A(3 classes)
`3 label`

- 1x1x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=92.66%(3x1x1)(no pretrained from imagenet)
2021.08.06
[TDN__high_events_RGB_resnet18_avg_segment1_e100](log/TDN__high_events_RGB_resnet18_avg_segment1_e100_True/log.txt)
`08/06`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 3`


`Overall Prec@1 92.66% Prec@5 100.00%`[3x1x1]
- val `93.1818` with [1x1x1](resnet18)
- test `92.09` with [3x1x1](resnet18)


- 3x1x1(Frames*crops*clips)
### TDN_resnet18_lr(0.01)_epoch(100)-top1=90.96%(3x1x1)(no pretrained from imagenet)
2021.08.06
[TDN__high_events_RGB_resnet18_avg_segment3_e100](log/TDN__high_events_RGB_resnet18_avg_segment3_e100_True/log.txt)
[graph](log/TDN__high_events_RGB_resnet18_avg_segment3_e100_True/events.out.tfevents.1628221674.workspace-job-5fadedfbbf46a9bf616072f7-xqd6z)
`08/06`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 3`


`Overall Prec@1 90.96% Prec@5 100.00%`[3x1x1]
- val `91.47727` with [3x1x1](resnet18)
- test `90.96` with [3x3x1](resnet18)



- 1x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=93.22%(5x1x1)(pretrained from imagenet)
2021.08.06
[log/TDN__high_events_RGB_resnet50_avg_segment1_e100](log/TDN__high_events_RGB_resnet50_avg_segment1_e100_True/log.txt)
[args](log/TDN__high_events_RGB_resnet50_avg_segment1_e100_True/args.txt)
[graph](log/TDN__high_events_RGB_resnet50_avg_segment1_e100_True/events.out.tfevents.1628230259.workspace-job-5fadedfbbf46a9bf616072f7-xqd6z)
`08/06`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 3`


`Overall Prec@1 92.66% Prec@5 100.00%`[1x1x1]
`Overall Prec@1 93.22% Prec@5 100.00%`[5x1x1]
- val `89.2045` with [1x1x1](resnet50)
- test `92.09` with [3x3x1](resnet50)


- 3x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=93.22%(3x1x1)(pretrained from imagenet)
2021.08.06
[TDN__high_events_RGB_resnet50_avg_segment3_e100](log/TDN__high_events_RGB_resnet50_avg_segment3_e100_True/log.txt)
[args](log/TDN__high_events_RGB_resnet50_avg_segment3_e100_True/args.txt)
[graph](log/TDN__high_events_RGB_resnet50_avg_segment3_e100_True/events.out.tfevents.1628234257.workspace-job-5fadedfbbf46a9bf616072f7-xqd6z)
`08/06`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 3`


`Overall Prec@1 93.22% Prec@5 100.00%`[3x1x1]
`Overall Prec@1 89.27% Prec@5 100.00%`[5x1x1]
- val `93.75` with [3x1x1](resnet50)
- test `92.66` with [3x3x1](resnet50)


- 5x1x1(Frames*crops*clips)
### TDN_resnet50_lr(0.01)_epoch(100)-top1=93.22%(3x1x1)(pretrained from imagenet)
2021.08.06
[TDN__high_events_RGB_resnet18_avg_segment1_e100](log/TDN__high_events_RGB_resnet18_avg_segment1_e100_True/log.txt)
`08/06`

`CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py  high_events \`
`--archs='resnet18' --weights  --test_segments=1 \`
`--test_crops=1 --clip_index 0 --batch_size=64 --full_res --output_dir ./result  \`
`-j 4 --topk 2`
`python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result --topk 3`


`Overall Prec@1 93.22% Prec@5 100.00%`[3x1x1]
`Overall Prec@1 89.27% Prec@5 100.00%`[5x1x1]
- val `93.75` with [3x1x1](resnet50)
- test `92.66` with [3x3x1](resnet50)