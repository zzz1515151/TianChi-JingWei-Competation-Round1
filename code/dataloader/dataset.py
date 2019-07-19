import os
import errno
import sys
import csv
import torch
import random
import numpy as np
import torchvision
from .transform import *
from PIL import Image
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class jingwei_train_dataset(data.Dataset):
    def __init__(self, csv_root):

        self.image_list = []
        self.label_list = []

        with open(csv_root, 'r') as f:
            for info in f:
                info_tmp = info.strip(' ').split()
                self.image_list.append(info_tmp[0])
                self.label_list.append(info_tmp[1])

        assert len(self.image_list) == len(self.label_list)

        self.rotate = RandomRotate(4)
        self.flip = RandomFlip()        

        self.image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            lambda x: np.asarray(x),
            lambda x: x.astype(np.float32),            
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], #resnet
                                std = [0.229, 0.224, 0.225])
            ])

        self.label_transform = transforms.Compose([
            lambda x: np.asarray(x),          
            lambda x: x.astype(np.int64),           
            ])



    def __getitem__(self, index):

        image = Image.open(self.image_list[index])
        label = Image.open(self.label_list[index])        
        image, label = self.rotate(image, label)
        image, label = self.flip(image, label)       
        image = self.image_transform(image)
        label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.image_list)

