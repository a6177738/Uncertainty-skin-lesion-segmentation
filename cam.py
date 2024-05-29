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
def make_cam(img: Image.Image, mask: Image.Image, colormap: str = 'jet', line="1",alpha: float = 0.7):
    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = np.asarray(overlay)
    overlay = (255 * cmap((overlay) ** 2)[:, :, :3]).astype(np.uint8)
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
    return overlayed_img
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
    Net = model.get_model(pretrained="mocov2").cuda()
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    # net.cuda()
    Net.eval()
    path_model = "checkpoints\\WSSS\\moco-alpha-0.25-bs128.pth"
    Net.load_state_dict(torch.load(path_model))
    s = 0
    for truth_name in os.listdir(source_path+"ISIC-2017_Test_v2_Data"):
        #original_name  = truth_name.replace("_segmentation.png",".jpg").strip()

        original_file = source_path+"ISIC-2017_Test_v2_Data\\"
        original = cv2.imread(original_file+truth_name)
        original = cv2.resize(original,(448,448))
        image = normalize_fn(original).transpose((2,0,1))

        image = torch.from_numpy(image).unsqueeze(dim=0).cuda()
        fg_feats, bg_feats, ccam = Net(image,inference=True)
        cam = 1-torch.nn.functional.interpolate(ccam,(448,448),mode='bilinear').squeeze().detach().cpu().numpy()
        cv2.imwrite("data\\ISIC-2017\\test_gray_cam\\" + truth_name, cam*255)
        # color_cam = make_cam(to_pil_image(original),to_pil_image(cam,mode='F'),alpha=0.7,line=original_name)
        # color_cam = np.array(color_cam)
        # cv2.imwrite("C:\\Users\\dell\\Desktop\\pifujing\\data\\ISIC-2017\\color_cam\\" + truth_name, color_cam)





if __name__ == "__main__":
    evaluate("data\\ISIC-2017\\",0)