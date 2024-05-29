import torch
from torch.utils.data import DataLoader
import numpy as np
from model import ResFCN
from data_recycle18 import Data
import cv2
from torch.autograd import Variable

@torch.no_grad()
def evaluate(add, epoch):
    Net = ResFCN()
    # net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    Net.cuda()
    Net.eval()
    path_model = "check\\" + str(epoch) + ".pth"
    Net.load_state_dict(torch.load(path_model))
    dataset = Data(add)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8
    )
    np.set_printoptions(threshold=10000)

    for batch_i, (img, img_id) in enumerate(dataloader):
        img = Variable(img.float()).cuda()
        fea_map = Net(img).squeeze().cpu().detach().numpy()

        for b in range(img.shape[0]):
            if img.shape[0]==1:
                print(img_id)
                cv2.imwrite("E:\\pifujing\\data\ISIC-2018\\iteration1\\" + img_id[0],
                            fea_map * 255)
            else:
                cv2.imwrite("E:\\pifujing\\data\\ISIC-2018\\iteration1\\" + img_id[b],
                        fea_map[b, :, :] * 255)



if __name__ == "__main__":


    evaluate("E:\\pifujing\\data\\ISIC-2018\\", "56")