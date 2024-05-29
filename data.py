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
        self._label_p = os.path.join('%s', 'cam_iteration3\\', '%s')
        self._label_b = os.path.join('%s', 'train_b_result_cut\\', '%s')
        for line in os.listdir(root+"ISIC-2017_Training_Data"):
            self.ids.append((root, line.strip()))
        for line in os.listdir(root+"ISIC-2017_Training_Data"):
            self.gt_ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        gt_id = self.gt_ids[index]
        label_p = cv2.imread(self._label_p % gt_id,0)
        label_p = Freqcam(label_p,gt_id[1])
        label_b = 255-label_p

        img = cv2.resize(img,(224,224))

        label_p = cv2.resize(label_p,(224,224))/255
        label_b = cv2.resize(label_b,(224,224))/255

        mode = random.randint(0,2)   #0,2都包含
        img,label_p,label_b = aug(img,label_p,label_b,mode = mode)



        img = img.transpose(2,0,1)

        return (img, label_p,label_b)
    def __len__(self):
        return len(self.ids)