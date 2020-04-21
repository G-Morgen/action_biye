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

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class HMDB51Dataset(Dataset):

    def __init__(self,mode='valid',modality='rgb',split=0,frame_counts=10,step_size=2,sample_times=10,sample='dense',transform=None):
        
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
        self._sample_times = sample_times

        train, test = utils.read_hmdb_file(split)
        if self._mode == 'train':
            self.data = train
        elif self._mode == 'valid':
            self.data = test
        print(self._mode + ' video num: %d' % len(self.data))

    def _sample_indices(self, num_frames):
       
        begin_node = num_frames - self._frame_counts * self._step_size
        if begin_node > 0:
            offsets = np.multiply(list(range(self._frame_counts)), self._step_size) + np.random.randint(begin_node, size=1)
        elif num_frames > self._frame_counts:
            offsets = np.arange(self._frame_counts) + np.random.randint(num_frames - self._frame_counts)
        else:
            offsets = np.pad(np.arange(num_frames),[0,self._frame_counts - num_frames], 'edge')
        return offsets

    def read_img(self,img_path):

        dir_list = os.listdir(img_path)
        img_list = [os.path.join(img_path, i) for i in dir_list]
        img_list = sorted(img_list)
        l = len(img_list)
        index = self._sample_indices(l)

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
            u = Image.open(u_list[i]).convert('L')
            v = Image.open(v_list[i]).convert('L')
            frames.extend([u,v])
        
        return frames

    def __getitem__(self,index):
        
        path = self.data[index][0]
        label = int(self.data[index][1])
        ab_path = os.path.join("/home/zhujian/dataset/hmdb51_frames/",path[:-4])

        clip_list = []
        for _ in range(self._sample_times):

            if self._modality == 'rgb':
                ab_path = os.path.join("/home/zhujian/dataset/hmdb51_frames/",path[:-4])
                frames = self.read_img(ab_path)
            else:
                u_path = os.path.join("/home/zhujian/dataset/hmdb51_flow/u/",path.split('/')[-1][:-4])
                v_path = os.path.join("/home/zhujian/dataset/hmdb51_flow/v/",path.split('/')[-1][:-4])
                frames = self.read_flow(u_path, v_path)

            if self.transform:
                frames = self.transform(frames)
            h,w = frames.size(1), frames.size(2)
            frames = frames.view(self._frame_counts,-1,h,w).permute(1,0,2,3)
            clip_list.append(frames)

        sample = {}
        sample['label_num'] = label
        sample['data'] = torch.stack(clip_list,dim=0)

        return sample

    def __len__(self):
        return len(self.data)

def make_data(num_frames,batch_size,modality,model,sample='dense',interval=2,num_workers=4):

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

    valid_dataset = HMDB51Dataset(
        mode='valid',
        frame_counts=num_frames,
        modality=modality,
        step_size=interval,
        sample=sample,
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

    valid_dataset = DataLoaderX(
        valid_dataset,
        batch_size=batch_size, shuffle=False, num_workers= num_workers, pin_memory=True
    )
    print('valid dataset init successfully')
    
    return valid_dataset, classes

if __name__ == "__main__":
    import torchvision
    import time
    from matplotlib import pyplot as plt

    frames = 8
    a, _ = make_data(frames,4,modality='flow',model='resnet50')
    t = time.time()
    for idx,i in enumerate(a):
        print(i['data'].shape)
        