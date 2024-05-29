import torch
from  torch.utils.data import  DataLoader
import numpy as np
import model
import cv2
import os
import copy
from PIL import  Image
from  test_isic17 import  evalua
import torch.nn.functional as F
from data_test import Data
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
def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride
def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size, mode=mode, align_corners=align_corners)


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
normalize_fn = Normalize(imagenet_mean, imagenet_std)
scales = '1.0'
scales = [float(scale) for scale in scales.split(',')]



Net = model.get_model(pretrained="mocov2")

def evaluate(source_path,epoc0):
    Net.eval()
    Net.cuda()
    path_model = "checkpoints/WSSS/moco-alpha-0.25-bs128.pth"
    ckpt = torch.load(path_model)
    Net.load_state_dict(ckpt)
    ground_path = "ISIC-2017_Test_v2_Part1_GroundTruth\\"
    data_path = "ISIC-2017_Test_v2_Data\\"
    dataset = Data(source_path,data_path,ground_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )
    for batch_i, (img, img_name) in enumerate(dataloader):

        original = img.squeeze().numpy()
        image = original

        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).unsqueeze(dim=0).cuda()

        # inferenece
        _, _, cam = Net(image, inference=True)

        # postprocessing
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        ######################################################################
        cam = (1-cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR))*255
        cam = cv2.threshold(cam, 140, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(source_path+"result\\" + img_name[0], cam)
        print(img_name[0])

        # cam = 1-torch.from_numpy(cam)
        # a = float(torch.sum(cam) / (224 * 224))
        # S+=a
        # print(a)
        # zero1 = torch.zeros((cam.shape))
        # M1 = torch.where(cam > a, cam, zero1)
        # activation_map = M1.detach().numpy()

        #region = cv2.threshold(activation_map, 127, 255, cv2.THRESH_BINARY)[1]
        # cv2.imwrite(source_path+"region\\" + img_name[0], region)
        # region1 = cv2.imread(source_path+"region\\" + img_name[0], 0)
        # num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(region1, connectivity=8)
        #
        # max = 0
        # n = 500
        # if num_objects > 1:
        #     for i in range(0, num_objects):
        #         loc = np.where(labels == i)
        #         if activation_map[loc].mean()>0.1 and  stats[i][4]>300:
        #             if n>abs(loc[0].mean()-112)+abs(loc[1].mean()-112):
        #                 n = abs(loc[0].mean()-112)+abs(loc[1].mean()-112)
        #                 max = i
        #     for i in range(0, num_objects):
        #         if i != max:
        #             loc = np.where(labels == i)
        #             activation_map[loc] = 0
        #             region[loc] = 0
        #cv2.imwrite(source_path+"result\\" + img_name[0], region)

if __name__ == "__main__":
    #for i in range(100):
        evaluate("data\\ISIC-2017\\",1)
        evalua(1)