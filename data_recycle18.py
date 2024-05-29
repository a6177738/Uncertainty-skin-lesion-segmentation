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
        self._imgpath = os.path.join('%s', 'ISIC2018_Train\\', '%s')
        for line in os.listdir(root+"ISIC2018_Train"):
            self.ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(224,224))
        img = img.transpose(2,0,1)

        return (img, img_id[1])
    def __len__(self):
        return len(self.ids)