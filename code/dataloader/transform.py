import torch 
import numpy as np 
from PIL import Image 

class RandomFlip(object):
    def __call__(self, image, label):
        if np.random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return image, label


class RandomRotate(object):
    def __init__(self, degree):
        '''degree: 旋转角度'''
        self.degree = degree

    def __call__(self, image, label):
        rotate_degree = np.random.uniform(0, self.degree)
        image = image.rotate(rotate_degree * 90, Image.BILINEAR)
        label = label.rotate(rotate_degree * 90, Image.NEAREST)
        return image, label

