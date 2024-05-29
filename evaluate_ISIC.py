import torch
from  torch.utils.data import  DataLoader
import numpy as np
import model
import cv2
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
np.set_printoptions(200)
@torch.no_grad()
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

def evaluate(source_path,epoch):
    Net = model.get_model(pretrained="mocov2")
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    # net.cuda()
    Net.eval()
    path_model = "checkpoints/WSSS/moco-alpha-0.25-bs128.pth"
    Net.load_state_dict(torch.load(path_model, map_location='cpu'))
    s = 0
    for truth_name in os.listdir(source_path+"ISIC-2017_Test_v2_Part1_GroundTruth/"):
        original_name  = truth_name.replace("_segmentation.png",".jpg").strip()

        original_file = source_path+"ISIC-2017_Test_v2_Data/"
        original = cv2.imread(original_file+original_name)
        original = cv2.resize(original,(280,224))
        original = original[:,28:252,:]
        image = normalize_fn(original).transpose((2,0,1))

        image = torch.from_numpy(image).unsqueeze(dim=0)
        fg_feats, bg_feats, ccam = Net(image,inference=True)

        M = 1-torch.nn.functional.interpolate(ccam, (224, 224), mode='bilinear').squeeze()
        a = 0.4844#float(torch.sum(M) / (224 * 224))
        s += a
        zero1 = torch.zeros((M.shape))
        M1 = torch.where(M > a, M, zero1)
        activation_map = M1.detach().numpy()
        #rerecv2.imwrite("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/CCAM/cam/" + truth_name, activation_map*255)
        region = cv2.threshold(activation_map, 0.01, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data//ISIC-2017/CCAM/region1/" + truth_name, region)
        region1 = cv2.imread("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/CCAM/region1/" + truth_name, 0)
        num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(region1, connectivity=8)
        max = 0
        second = 0
        if num_objects > 1:
            for i in range(0, num_objects):
                if stats[i][4] > max:
                    max = stats[i][4]
            for i in range(0, num_objects):
                if stats[i][4] > second and stats[i][4]!=max:
                    second = stats[i][4]
            for i in range(0, num_objects):
                if stats[i][4] == max:
                    loc = np.where(labels == i)
                    if activation_map[loc].mean()<0.01:
                        max = second
                if stats[i][4] < max:
                    loc = np.where(labels == i)
                    activation_map[loc] = 0
                    region[loc] = 0

        cv2.imwrite("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/CCAM/s_cam1/"+truth_name,activation_map*255)
        cv2.imwrite("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/CCAM/result1/" + truth_name, region)
    print(s/600)

if __name__ == "__main__":
    evaluate("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/",0)