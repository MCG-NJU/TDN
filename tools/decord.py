# from decord import VideoReader
import decord
# from decord import cpu, gpu

vr = decord.VideoReader('/workspace/mnt/storage/kanghaidong/action_data/kinetics-400-encode/train/hurling_(sport)/CaWo4-KgEOM_000022_000032.mp4')
print(type(vr))
print('video frames:', len(vr))
# 1. the simplest way is to directly access frames
for i in range(len(vr)):
    # the video reader will handle seeking and skipping in the most efficient manner
    frame = vr[i]
    print(frame.shape) # (256,x)