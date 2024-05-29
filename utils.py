#划分高显著性中显著性低显著性区域
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import cm
#高中低显著性区域划分
def split_saliency(cam):
    ones = torch.ones(cam.shape)#置1
    zeros = torch.zeros(cam.shape)#用于置0

    high_region = torch.where(cam>=0.7,ones,zeros)*255
    low_region = torch.where(cam<=0.3,ones,zeros)*255

    mid_region = torch.where(cam<0.7,cam,zeros)
    mid_region = torch.where(mid_region>0.3,ones,zeros)*255

    return high_region,mid_region,low_region

#连通性检测
def connected(region):
    select_region = region.copy()

    num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)

    max = 0
    n = 500
    if num_objects > 1:
        for i in range(0, num_objects):
           loc = np.where(labels == i)
           if region[loc].mean()>1 and  stats[i][4]>30:
               if n>abs(loc[0].mean()-112)+abs(loc[1].mean()-112):
                   n = abs(loc[0].mean()-112)+abs(loc[1].mean()-112)
                   max = i
    for i in range(0, num_objects):
       if i != max:
           loc = np.where(labels == i)
           select_region[loc] = 0

    return select_region, n

def make_cam(img: Image.Image, mask: Image.Image, colormap: str = 'jet', line="1",alpha: float = 0.7):
    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = np.asarray(overlay)
    overlay = (255 * cmap((overlay) ** 2)[:, :, :3]).astype(np.uint8)
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
    return overlayed_img