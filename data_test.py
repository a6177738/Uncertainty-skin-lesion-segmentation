import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import PIL.Image
import random

class Data(Dataset):
    def __init__(self, root,data_path,ground_path,transforms="False"):
        self.root = root
        self._imgpath = os.path.join('%s', '%s')#
        self.ids = list()
        self.transforms = transforms
        for line in os.listdir(root+ground_path):
            self.ids.append((root+data_path, line.replace("_segmentation.png",".jpg").strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        img = cv2.resize(img, (224,224))

        return (img,img_id[1])
    def __len__(self):
        return len(self.ids)