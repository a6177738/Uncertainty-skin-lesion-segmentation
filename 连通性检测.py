import torch
from  torch.utils.data import  DataLoader
import numpy as np
import model
import cv2
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
from utils import *
np.set_printoptions(200)



def evaluate(source_path):
    for truth_name in os.listdir(source_path):
        if truth_name == ".DS_Store":
            continue
        img = cv2.imread(source_path+"/"+truth_name,0)
        img = cv2.resize(img,(224,224))
        fore_ground,n = connected(img)

        cv2.imwrite("/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/对比数据/ccam_connec/" + truth_name, fore_ground)
        print(truth_name)

if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/对比数据/ccam_b")