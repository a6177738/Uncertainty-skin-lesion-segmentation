import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random


class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = list()
        self._imgpath = os.path.join('%s', 'ISIC-2017_Training_Data\\', '%s')
        for line in os.listdir(root+"ISIC-2017_Training_Data"):
            self.ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(224,224))
        img_resize = img.transpose(2,0,1)

        return (img_resize, img_id[1])
    def __len__(self):
        return len(self.ids)