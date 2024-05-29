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
        self.gt_ids = list()
        self._imgpath = os.path.join('%s', 'ISIC2018_Test\\', '%s')
        self._label = os.path.join('%s', 'ISIC2018_Test_GroundTruth\\', '%s')
        for line in os.listdir(root+"ISIC2018_Test_GroundTruth"):
            self.ids.append((root, line.replace("_segmentation.png",".jpg").strip()))
        for line in os.listdir(root+"ISIC2018_Test_GroundTruth"):
            self.gt_ids.append((root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        gt_id = self.gt_ids[index]

        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        label = cv2.imread(self._label % gt_id,0)



        img = cv2.resize(img,(224,224))

        label = cv2.resize(label,(224,224))/255
        label = cv2.threshold(label,0.5,1,cv2.THRESH_BINARY)[1]


        img_resize = img.transpose(2,0,1)

        return (img_resize, label, img_id[1])
    def __len__(self):
        return len(self.ids)