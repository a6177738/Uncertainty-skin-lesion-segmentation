import numpy as np
from torch.utils.data import Dataset
import torch
import os
import cv2
import random
from PIL import Image
from transform import *
from freqcam import Freqcam
def aug(img,label_p,label_b,mode=0):
   return random_flip(img,label_p,label_b,mode)


class Data(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = list()
        self.gt_ids = list()
        self._imgpath = os.path.join('%s', 'Dataset\\', '%s')
        for line in os.listdir(root+"result"):
            image_name = line.replace(".png","")
            im_path = image_name+"\\"+image_name+"_Dermoscopic_Image\\"+image_name+".bmp"
            gt_path = "iteration4\\"+line
            self.ids.append((root, im_path))
            self.gt_ids.append((root+gt_path))

    def __getitem__(self, index):

        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        gt_id = self.gt_ids[index]
        label = cv2.imread(gt_id,0)

        img = cv2.resize(img,(224,224))

        T = 172

        label_p = Freqcam(label,T)

        label_no_p = (255-label_p)/255.0*label #非前景
        label_b_highs = cv2.threshold(label_no_p,T,255,cv2.THRESH_BINARY)[1] #高显著性背景
        label_b_lows = 255-cv2.threshold(label,T/2,255,cv2.THRESH_BINARY)[1] #低显著性背景
        label_b = label_b_lows+label_b_highs

        label_p = cv2.threshold(label_p,172,255,cv2.THRESH_BINARY)[1]

        label_p = cv2.resize(label_p,(224,224))/255
        label_b = cv2.resize(label_b,(224,224))/255
        img, label_p, label_b = aug(img, label_p, label_b)

        img_resize = img.transpose(2,0,1)

        return (img_resize, label_p, label_b)
    def __len__(self):
        return len(self.ids)