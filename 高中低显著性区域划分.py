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
class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = np.asarray(image)
        norm_image = np.empty_like(image, np.float32)

        norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        return norm_image
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize_fn = Normalize(imagenet_mean, imagenet_std)

@torch.no_grad()
def evaluate(source_path):
    Net = model.get_model(pretrained="mocov2")
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    # net.cuda()
    Net.eval()
    path_model = "checkpoints/WSSS/moco-alpha-0.25-bs128.pth"
    write_source = "/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/画图所需数据/"
    Net.load_state_dict(torch.load(path_model, map_location='cpu'))
    s = 0
    for truth_name in os.listdir(source_path+"ISIC-2017_Training_Part1_GroundTruth/"):
        original_name  = truth_name.replace("_segmentation.png",".jpg").strip()

        original_file = source_path+"ISIC-2017_Training_Data/"
        original = cv2.imread(original_file+original_name)
        original = cv2.resize(original,(224,224))
        image = normalize_fn(original).transpose((2,0,1))

        image = torch.from_numpy(image).unsqueeze(dim=0)
        fg_feats, bg_feats, ccam = Net(image,inference=True)

        cam = 1-torch.nn.functional.interpolate(ccam, (224, 224), mode='bilinear').squeeze().detach()
        cv2.imwrite(write_source +"gray_cam/"+ truth_name, cam.numpy()*255)
        print(truth_name)
        #阈值筛选高显著性区域，中显著性区域，低显著性区域
        # high_region,mid_region,low_region= split_saliency(cam)
        #
        # high_region, mid_region, low_region = high_region.numpy(),mid_region.numpy(),low_region.numpy()
        # cv2.imwrite(write_source +"high_region/"+ truth_name, high_region)
        # cv2.imwrite(write_source +"mid_region/"+ truth_name, mid_region)
        # cv2.imwrite(write_source +"low_region/"+ truth_name, low_region)
    print("s",s/600)

if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/")