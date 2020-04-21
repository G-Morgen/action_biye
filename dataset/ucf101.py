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
import decord
import logging 

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class UCF101Dataset(Dataset):

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
        self.transform = transform[0 if modality == 'rgb' else 1]
        if self._modality == 'fusion':
            self.rgb_transform = transform[0]
            self.flow_transform = transform[1]
        self._sample = sample

        train, test = utils.read_file(split)
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
                offsets = np.sort(np.random.randint(num_frames + 1, size=self._frame_counts))
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
                offsets = np.pad(np.arange(num_frames),[0,self._frame_counts - num_frames], 'edge')
            return offsets

    def read_img(self,img_path):

        img_list = glob.glob(os.path.join(img_path,'*'))
        img_list = sorted(img_list)

        l = len(img_list)
        index = self._sample_indices(l)
        index = np.int32(index)
        frames = []
        for i in index:
            img = Image.open(img_list[i]).convert('RGB')
            frames.append(img)
        return frames

    def read_flow(self,u_path,v_path):
        
        u_list = sorted(glob.glob(os.path.join(u_path,'*')))
        v_list = sorted(glob.glob(os.path.join(v_path,'*')))
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
            ab_path = os.path.join("/home/zhujian/dataset/UCF-101_frames/",path[:-4])
            frames = self.read_img(ab_path)
        elif self._modality == 'flow' :
            u_path = os.path.join("/home/zhujian/dataset/ucf101_tvl1_flow/tvl1_flow/u/",path[:-4])
            v_path = os.path.join("/home/zhujian/dataset/ucf101_tvl1_flow/tvl1_flow/v/",path[:-4])
            frames = self.read_flow(u_path, v_path)
        elif self._modality == 'fusion':
            ab_path = os.path.join("/home/zhujian/dataset/UCF-101_frames/",path[:-4])
            rgb = self.read_img(ab_path)
            u_path = os.path.join("/home/zhujian/dataset/ucf101_tvl1_flow/tvl1_flow/u/",path[:-4])
            v_path = os.path.join("/home/zhujian/dataset/ucf101_tvl1_flow/tvl1_flow/v/",path[:-4])
            flow = self.read_flow(u_path, v_path)

        if self._modality != 'fusion':
            if self.transform:
                frames = self.transform(frames)

            sample = {}
            sample['path'] = path
            sample['label_num'] = label
            if self._sample == 'sparse':
                sample['data'] = frames
            else:
                h,w = frames.size(1), frames.size(2)
                sample['data'] = frames.view(self._frame_counts,-1,h,w).permute(1,0,2,3)
        else:
            if self.transform:
                rgb = self.rgb_transform(rgb)
                flow = self.flow_transform(flow)
            
            sample = {}
            sample['path'] = path
            sample['label_num'] = label
            if self._sample == 'sparse':
                sample['rgb'] = rgb
                sample['flow'] = flow
            else:
                h,w = frames.size(1), frames.size(2)
                sample['rgb'] = rgb.view(self._frame_counts,-1,h,w).permute(1,0,2,3)
                sample['flow'] = flow.view(self._frame_counts,-1,h,w).permute(1,0,2,3)

        return sample

    def __len__(self):
        return len(self.data)

def get_train_aug(sample, mode='train'):
    if mode != 'train':
        return GroupCenterCrop(224)

    if sample == 'dense':
        return GroupRandomCrop(224)
    else:
        return GroupMultiScaleCrop(224, [1, .875, .75, .66])
    
    

def make_data(num_frames,batch_size,modality,model,split=0,sample='sparse',interval=2,num_workers=4):

    if 'bn_inception' == model:
        rgb_mean = [104, 117, 128]
        rgb_std = [1, 1, 1]
        flow_mean = [128]
        flow_std = [1]
    else:
        rgb_mean=[0.485, 0.456, 0.406]
        rgb_std=[0.229, 0.224, 0.225]
        flow_mean = [0.5]
        flow_std = [np.mean(rgb_std)]

    def rgb_transform(mode='train'):
        return torchvision.transforms.Compose([
                get_train_aug(sample, mode),
                GroupRandomHorizontalFlip(is_flow=False),
                Stack(roll= model == 'bn_inception'),
                ToTorchFormatTensor(div= model != 'bn_inception'),
                GroupNormalize(
                    mean=rgb_mean,
                    std=rgb_std
                )
            ])

    def flow_transform(mode='train'):
        return torchvision.transforms.Compose([
            get_train_aug(sample, mode),
            GroupRandomHorizontalFlip(is_flow=True),
            Stack(roll= model == 'bn_inception'),
            ToTorchFormatTensor(div= model != 'bn_inception'),
            GroupNormalize(
                mean=flow_mean,
                std=flow_std
            )
        ])

    train_dataset = UCF101Dataset(
        mode='train',
        frame_counts=num_frames,
        sample=sample,
        split=split,
        modality=modality,
        step_size=interval,
        transform=[rgb_transform(), flow_transform()]
    )

    valid_dataset = UCF101Dataset(
        mode='valid',
        frame_counts=num_frames,
        sample=sample,
        split=split,
        modality=modality,
        step_size=interval,
        transform=[rgb_transform('valid'), flow_transform('valid')]
    )

    classes = 101

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

    frames = 5
    a, b, _ = make_data(frames,4,modality='flow', model='bn_inception', sample='dense')
    t = time.time()
    for idx,i in enumerate(a):
       print(i['data'].shape)

    # run_check_dense_rgb()

    # data = a['data'].view((10,3,224,224))
    
    # img = data.permute(1,2,3,0).cpu().numpy()[0]
    
    # for i in range(5):
    #     a = img[...,i*2]
    #     plt.subplot(1,5,i+1)
    #     plt.imshow(a,cmap=plt.cm.gray)
    # plt.show()