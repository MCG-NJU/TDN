# # something
# CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py something \
# --archs='resnet50' --weights somting_8x1x1_best.pth.tar  --test_segments=8  \
# --test_crops=1 --batch_size=32 --output_dir ./result -j 4 --clip_index=0

# python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result

# somethingv2
# CUDA_VISIBLE_DEVICES=0,1 python test_models_center_crop.py somethingv2 \
# --archs='resnet50' --weights checkpoint/TDN__somethingv2_RGB_resnet50_avg_segment8_e100/best.pth.tar  --test_segments=8  \
# --test_crops=1 --batch_size=32 --output_dir ./result -j 4 --clip_index=0

# python pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir ./result

# kinetics
CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics_frame \
--archs='resnet50' --weights tdn-models/TDN__kinetics_frame_RGB_resnet50_avg_segment8_e100_best.pth.tar  --test_segments=8 \
--test_crops=3 --clip_index 3 --batch_size=1 --full_res --output_dir ./result  \
-j 4
python pkl_to_results.py --num_clips 1 --test_crops 3 --output_dir ./result