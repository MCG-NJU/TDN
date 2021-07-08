# # something
# CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py something \
# --archs='resnet50' --weights somting_8x1x1_best.pth.tar  --test_segments=8  \
# --test_crops=1 --batch_size=32 --output_dir ./result -j 4 --clip_index=0

# python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result

# somethingv2
CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py somethingv2 \
--archs='resnet50' --weights checkpoint/TDN__somethingv2_RGB_resnet50_avg_segment8_e60/best.pth.tar  --test_segments=8  \
--test_crops=1 --batch_size=32 --output_dir ./result -j 4 --clip_index=0

python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result


# CUDA_VISIBLE_DEVICES=0,1 python3 test_models_three_crops.py  kinetics \
# --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
# --test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir ./result  \
# -j 4 --clip_index ./clip_index
# python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir ./result