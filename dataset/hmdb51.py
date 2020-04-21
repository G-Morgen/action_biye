import torch 
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import glob
import cv2
import json 

import sys
try:
    sys.path.append('.')
    from dataset.transforms import *
    from dataset import utils
except:
    from transforms import *
    import utils

from memory_profiler import profile
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class HMDB51Dataset(Dataset):

    def __init__(self,split=0,mode='train',modality='rgb',frame_counts=10,step_size=2,sample='sparse',transform=None):
        
        if sample not in ['sparse','dense']:
            raise ValueError('no that sample methods %s' % sample)
        if mode not in ['train','valid','test']:
            raise ValueError('no that mode %s' % mode)

        self._mode = mode
        self._split = split
        self._modality = modality
        self._step_size = step_size
        self._frame_counts = frame_counts
        self.transform = transform
        self._sample = sample

        train, test = utils.read_hmdb_file(split)
        if self._mode == 'train':
            self.data = train
        elif self._mode == 'valid':
            self.data = test
        print(self._mode + ' video num: %d' % len(self.data))


    def _sample_indices(self, num_frames):

        if self._sample == 'sparse':
            average_duration = (num_frames) // self._frame_counts
            if average_duration > 0:
                offsets = np.multiply(list(range(self._frame_counts)), average_duration) + np.random.randint(average_duration, size=self._frame_counts)
            elif num_frames > self._frame_counts:
                offsets = np.sort(np.random.randint(num_frames, size=self._frame_counts))
            else:
                offsets = np.zeros((self._frame_counts,))
            return offsets 

        else:
            begin_node = num_frames - self._frame_counts * self._step_size
            if begin_node > 0:
                offsets = np.multiply(list(range(self._frame_counts)), self._step_size) + np.random.randint(begin_node, size=1)
            elif num_frames > self._frame_counts:
                offsets = np.arange(self._frame_counts) + np.random.randint(num_frames - self._frame_counts)
            else:
                offsets = np.zeros((self._frame_counts,))
            
            return offsets


    def read_img(self,img_path):

        dir_list = os.listdir(img_path)
        img_list = [os.path.join(img_path, i) for i in dir_list]
        img_list = sorted(img_list)
        
        l = len(img_list)
        index = self._sample_indices(l-1)
        index = np.int32(index)
        frames = []
        for i in index:
            img = Image.open(img_list[i]).convert('RGB')
            frames.append(img)
        
        return frames
    
    def read_flow(self,u_path,v_path):
        
        u_list = sorted([os.path.join(u_path,i) for i in os.listdir(u_path)])
        v_list = sorted([os.path.join(v_path,i) for i in os.listdir(v_path)])
        l = len(u_list)
        index = self._sample_indices(l-5)
        index = np.int32(index)

        frames = []
        for i in index:
            if self._sample == 'sparse':
                for j in range(i,i+5):
                    u = Image.open(u_list[j]).convert('L')
                    v = Image.open(v_list[j]).convert('L')
                    frames.extend([u,v])
            else:
                u = Image.open(u_list[i]).convert('L')
                v = Image.open(v_list[i]).convert('L')
                frames.extend([u,v])
        return frames

    def __getitem__(self,index):
        
        path = self.data[index][0]
        label = int(self.data[index][1])
        if self._modality == 'rgb':
            ab_path = os.path.join("/home/zhujian/dataset/hmdb51_frames/",path[:-4])
            frames = self.read_img(ab_path)
        else:
            u_path = os.path.join("/home/zhujian/dataset/hmdb51_flow/u/",path.split('/')[-1][:-4])
            v_path = os.path.join("/home/zhujian/dataset/hmdb51_flow/v/",path.split('/')[-1][:-4])
            frames = self.read_flow(u_path, v_path)

        if self.transform:
            frames = self.transform(frames)

        sample = {}
        sample['label_num'] = label
        if self._sample == 'sparse':
            sample['data'] = frames
        else:
            h,w = frames.size(1), frames.size(2)
            sample['data'] = frames.view(self._frame_counts,-1,h,w).permute(1,0,2,3)

        return sample

    def __len__(self):
        return len(self.data)


def get_train_aug(sample):
    if sample == 'dense':
        return GroupRandomCrop(224)
    else:
        return GroupMultiScaleCrop(224, [1, .875, .75, .66])

def make_data(num_frames,batch_size,modality,model,split=0,sample='sparse',interval=2,num_workers=4):

    if 'bn_inception' == model:
        mean = [104, 117, 128]
        std = [1, 1, 1]
        if modality == 'flow':
            mean = [128]
            std = [1]
    else:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if modality == 'flow':
            mean = [0.5]
            std = [np.mean(std)]

    train_dataset = HMDB51Dataset(
        mode='train',
        frame_counts=num_frames,
        sample=sample,
        split=split,
        modality=modality,
        step_size=interval,
        transform=torchvision.transforms.Compose([
            get_train_aug(sample),
            GroupRandomHorizontalFlip(is_flow=modality == 'flow'),
            Stack(roll= model == 'bn_inception'),
            ToTorchFormatTensor(div= model != 'bn_inception'),
            GroupNormalize(
                mean=mean,
                std=std
            )
            ]
        )
    )

    valid_dataset = HMDB51Dataset(
        mode='valid',
        frame_counts=num_frames,
        sample=sample,
        split=split,
        modality=modality,
        step_size=interval,
        transform=torchvision.transforms.Compose([
            GroupCenterCrop(224),
            Stack(roll= model == 'bn_inception'),
            ToTorchFormatTensor(div= model != 'bn_inception'),
            GroupNormalize(
                mean=mean,
                std=std
            )
            ]
        )
    )

    classes = 51

    train_dataset = DataLoaderX(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    print('train dataset init successfully')
    valid_dataset = DataLoaderX(
        valid_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    print('valid dataset init successfully')
    
    return train_dataset, valid_dataset, classes


if __name__ == "__main__":
    import torchvision
    import time
    from matplotlib import pyplot as plt

    frames = 3
    a, _, _ = make_data(frames,4,'rgb','bn_inception',sample='dense',split=0)
    t = time.time()
    for idx,i in enumerate(a):
        # plt.show()
        print(i['data'].shape, i['label_num'],time.time() - t)
        t = time.time()
        