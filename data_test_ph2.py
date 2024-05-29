import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random
from PIL import Image
from transform import *
def aug(img,label_p,label_b,mode=0):
   return random_flip(img,label_p,label_b,mode)


class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = list()
        self.gt_ids = list()
        self._imgpath = os.path.join('%s', 'Dataset\\', '%s')
        for line in os.listdir(root+"result"):
            line = line.replace(".png","")
            im_path = line+"\\"+line+"_Dermoscopic_Image\\"+line+".bmp"
            gt_path = line+"\\"+line+"_lesion\\"+line+"_lesion.bmp"
            self.ids.append((root, im_path))
            self.gt_ids.append((root, gt_path))

    def __getitem__(self, index):

        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        gt_id = self.gt_ids[index]
        label = cv2.imread(self._imgpath % gt_id,0)

        img = cv2.resize(img,(224,224))

        label = cv2.resize(label,(224,224))/255
        label = cv2.threshold(label,0.5,1,cv2.THRESH_BINARY)[1]


        img_resize = img.transpose(2,0,1)

        return (img_resize, label, img_id[1])
    def __len__(self):
        return len(self.ids)