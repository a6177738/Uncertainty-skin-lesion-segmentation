import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random
from PIL import Image
from freqcam import Freqcam
from transform import *
def aug(img,label_p,label_b,mode=0):
   return random_flip(img,label_p,label_b,mode)


class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = list()
        self.gt_ids = list()
        self._imgpath = os.path.join('%s', 'ISIC-2017_Training_Data\\', '%s')
        self._label_p = os.path.join('%s', 'train_cam\\', '%s')
        for line in os.listdir(root+"train_cam"):
            self.ids.append((root, line.replace("_segmentation.png",".jpg").strip()))
            self.gt_ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        gt_id = self.gt_ids[index]
        label_p = cv2.imread(self._label_p % gt_id,0)
        label_p = 255 - label_p
        label_p = Freqcam(label_p,0.3*255)

        img = cv2.resize(img,(224,224))

        label_p = cv2.resize(label_p,(224,224))/255




        img = img.transpose(2,0,1)

        return (img, label_p)
    def __len__(self):
        return len(self.ids)