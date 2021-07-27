cd /workspace/mnt/storage/kanghaidong/new_video_project/video_project/TDN
sh install.sh

cd /root/.cache
mkdir torch
cd torch
mkdir checkpoints
cp /workspace/mnt/storage/kanghaidong/khd-project/awesome_work_project/work_gitlab/fire-events/pre-model/resnet50-19c8e357.pth /root/.cache/torch/checkpoints
cd /workspace/mnt/storage/kanghaidong/new_video_project/video_project/TDN

# # kinetics
# python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
#         main.py  ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
#         --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 8 \
#         --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --decoded_type decord

# something
python -m torch.distributed.launch --master_port 12348 --nproc_per_node=2 \
            main.py  high_events  RGB --arch resnet18 --num_segments 1 --gd 20 --lr 0.02 \
            --lr_scheduler step --lr_steps  50 75 90 --epochs 100 --batch-size 4 \
            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --topk 2