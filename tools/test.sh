CUDA_VISIBLE_DEVICES=0 python test_models_center_crop.py kinetics \
--archs='resnet50' --weights best.pth.tar  --test_segments=8  \
--test_crops=1 --batch_size=32 --output_dir ./result -j 4 --clip_index=0

python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result


# CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics \
# --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
# --test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir ./result  \
# -j 4 --clip_index ./clip_index
# python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir ./result