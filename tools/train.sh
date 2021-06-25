python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
        main.py  ucf101 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
        --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 8 \
        --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb --decoded_type decord