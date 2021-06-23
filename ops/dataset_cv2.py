# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch.utils.data as data
# import decord
from PIL import Image
import os
import numpy as np
from numpy.random import randint
import cv2

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])


class TSNDataSet_Cv2(data.Dataset):
    def __init__(self, dataset, root_path, list_file,
                 num_segments=3, new_length=5, clip_index=0, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.clip_index = clip_index
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.dataset = dataset
        self.remove_missing = remove_missing
        self.I3D_sample = False  # using dense sample as I3D
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]

        if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list):
        if not self.I3D_sample : # TSN uniformly sampling for TDN
            if((video_list - self.new_length + 1) < self.num_segments):
                average_duration = (video_list - 5 + 1) // (self.num_segments)
            else:
                average_duration = (video_list - self.new_length + 1) // (self.num_segments)
            offsets = []
            if average_duration > 0:
                offsets += list(np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,size=self.num_segments))
            elif video_list > self.num_segments:
                if((video_list - self.new_length + 1) >= self.num_segments):
                    offsets += list(np.sort(randint(video_list - self.new_length + 1, size=self.num_segments)))
                else:
                    offsets += list(np.sort(randint(video_list - 5 + 1, size=self.num_segments)))
            else:
                offsets += list(np.zeros((self.num_segments,)))
            offsets = np.array(offsets)
            return offsets + 1
        elif self.dataset == 'kinetics' and self.I3D_sample and (not self.dense_sample) : # i3d type sampling for training
            sample_pos = max(1, 1 + video_list - self.new_length - 64)
            t_stride = 64 // self.num_segments
            start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx1) % video_list for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.dense_sample:  # i3d dense sample for test
            sample_pos = max(1, 1 + video_list - self.new_length - 64)
            t_stride = 64 // self.num_segments
            start_idx1 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx2 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx3 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx4 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx5 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx6 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx7 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx8 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx9 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            start_idx10 = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx1) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx2) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx3) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx4) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx5) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx6) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx7) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx8) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx9) % video_list for idx in range(self.num_segments)]+[(idx * t_stride + start_idx10) % video_list for idx in range(self.num_segments)]
            return np.array(offsets) + 1


    def _get_val_indices(self, video_list):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + video_list - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % video_list for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if video_list > self.num_segments + self.new_length - 1:
                tick = (video_list - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, video_list):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + video_list - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % video_list for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if video_list > self.num_segments + self.new_length - 1:
                tick = (video_list - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        
        if('something' in self.dataset): 
            decode_boo = False
            video_list = os.listdir(record.path)
        
        else:
            decode_boo = True
            try:
                directory = record.path
                # if directory[-4:] != ".avi": # mp4,avi;but it so bad.
                #     video_path = directory+".avi"
                # elif directory[-4:] != ".mp4":
                #     video_path = directory+".mp4"
                # else:
                video_path = directory
                # print(video_path,index)
                # video_list = decord.VideoReader(video_path)
                video_list = cv2.VideoCapture(video_path)
                num_frames = int(video_list.get(cv2.CAP_PROP_FRAME_COUNT))
                
            except UnicodeDecodeError:
                decode_boo = False
                video_list = os.listdir(record.path)
        
        
        if not self.test_mode:
            if self.I3D_sample :
                segment_indices = self._sample_indices(num_frames) 
            else:
                segment_indices = self._sample_indices(num_frames) if self.random_shift else self._get_val_indices(num_frames) 
        else:
            if self.dataset == 'kinetics':
                segment_indices = self._sample_indices(num_frames)
            else:
                segment_indices = self._get_test_indices(num_frames)

        
        return self.get(record,video_list,num_frames, segment_indices,decode_boo)

    def get(self, record,video_list,num_frames, indices,decode_boo):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(0,self.new_length,1):
                if(decode_boo):
                    video_list.set(cv2.CAP_PROP_POS_FRAMES, p-1)
                    _,cv2_im = video_list.read()
                    if cv2_im is None:
                        print(record.path)

                    # seg_imgs = [Image.fromarray(video_list[p-1].asnumpy()).convert('RGB')]
                    seg_imgs = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
                    seg_imgs = Image.fromarray(cv2_im)
                else:
                    seg_imgs = self._load_image(record.path,p)
                images.append(seg_imgs)
                if((num_frames-self.new_length*1+1)>=8):
                    if p < (num_frames):
                        p += 1
                else:
                    if p < (num_frames):
                        p += 1

        process_data, record_label = self.transform((images,record.label))

        return process_data, record_label

    def __len__(self):
        return len(self.video_list)
