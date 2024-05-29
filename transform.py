import math
import numpy as np
import random
from PIL import Image
import PIL
 
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
 
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask_p=None,mask_b=None):
        for t in self.transforms:
            image, mask_p,mask_b = t(image, mask_p,mask_b)
        return (image, mask_p,mask_b)
 
class Resize(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target=None):
        image = F.resize(image, [self.size,self.size])
        if target is not None:
            target = F.resize(target, [self.size,self.size], interpolation=F.InterpolationMode.NEAREST)
        return image, target
 
 
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, target_p=None,target_b=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target_p is not None:
                target_p = F.hflip(target_p)
            if target_b is not None:
                target_b = F.hflip(target_b)
        return image, target_p, target_b
 
class RandomCrop(object):
    def __init__(self, size,flip):
        self.size = size
        self.flip_prob = flip
 
    def __call__(self, image, target_p,target_b):

        if random.random() < self.flip_prob:
            image = image.resize((256, 256))
            target_p = target_p.resize((256, 256))
            target_b = target_b.resize((256, 256))
            crop_params = T.RandomCrop.get_params(image, self.size)
            image = F.crop(image, *crop_params)
            if target_p is not None:
                target_p = F.crop(target_p, *crop_params)
            if target_b is not None:
                target_b = F.crop(target_b, *crop_params)
        return image, target_p,target_b
 
class CenterCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target
 
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
 
class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value
 
    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target
 
class ToTensor(object):
    def __call__(self, image, target_p,target_b):
        image = F.to_tensor(image)
        if target_p is not None:
            target_p = torch.as_tensor(np.array(target_p), dtype=torch.float32)
        if target_b is not None:
            target_b = torch.as_tensor(np.array(target_b), dtype=torch.float32)
        return image, target_p,target_b

class ToPIL(object):
    def __call__(self, image, target):
        image = Image.fromarray(image)
        target = Image.fromarray(target)
        return image, target

def random_flip(img,lp,lb, mode=1):
            """
            随机翻转
            :param img:
            :param model: 1=水平翻转 / 0=垂直 / -1=水平垂直
            :return:
            """
            mode = random.randint(0,2)
            assert mode in (0, 1, 2), "mode is not right"
            flip = np.random.choice(2) * 2 - 1  # -1 / 1
            if mode == 1:
                img = img[:, ::flip, :].copy()
                lp = lp[:,::flip].copy()
                lb = lb[:,::flip].copy()
            elif mode == 0:
                img = img[::flip, :, :].copy()
                lp = lp[::flip,:].copy()
                lb = lb[::flip,:].copy()
            elif mode == 2:
                img = img[::flip, ::flip, :].copy()
                lp = lp[::flip,::flip].copy()
                lb = lb[::flip,::flip].copy()

            return img,lp,lb

