import torch
from  torch.utils.data import  DataLoader
import numpy as np
from  model import ResFCN
from recycle_data import Data
import cv2
from torch.autograd import Variable
from freqcam import Freqcam

@torch.no_grad()
def evaluate(add,epoch):
    Net = ResFCN()
    #net = torch.nn.DataParallel(net, device_ids=list(range(0,2)))
    Net.cuda()
    Net.eval()
    path_model = "save_check\\"+str(epoch)+".pth"
    Net.load_state_dict(torch.load(path_model))
    dataset = Data(add)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers = 8
    )
    np.set_printoptions(threshold=10000)

    for batch_i, (img ,img_id) in enumerate(dataloader):
        img = Variable(img.float()).cuda()
        fea_map = Net(img).squeeze().cpu().detach().numpy()

        #pre = cv2.threshold(fea_map, 0.9, 1, cv2.THRESH_BINARY)[1]
        for b in range(fea_map.shape[0]):
            #out = Freqcam(fea_map[b,:,:],img_id[b])
            print(fea_map[b].shape)
            cv2.imwrite(add+"cam_iteration3\\"+img_id[b],fea_map[b]*255)

if __name__ == "__main__":
    #for i in range(130):

        evaluate("C:\\Users\\dell\\Desktop\\pifujing\\data\\ISIC-2017\\","187_3")