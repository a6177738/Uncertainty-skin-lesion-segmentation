import torch
from  torch.utils.data import  DataLoader
import numpy as np
from  model import ResFCN
from data_test18 import Data
import cv2
from torch.autograd import Variable
from freqcam import Freqcam

def dice_coefficient(y_pred, y_true):
    """dice coefficient 2nt/na + nb."""
    y_true = y_true
    y_pred = y_pred
    union = y_true * y_pred
    dice = 2 * np.sum(union) / (np.sum(y_true) + np.sum(y_pred))
    return dice
def jac_coefficient(y_pred, y_true):
    """dice coefficient 2nt/na + nb."""
    y_true = y_true
    y_pred = y_pred
    union = y_true * y_pred
    Jac =  np.sum(union) / (np.sum(y_true) + np.sum(y_pred)-np.sum(union))
    return Jac
def SPE(y_pred, y_true):
    """SPE TN/(TN+FP)."""
    true = 1-y_true
    pred = 1-y_pred
    union = true * pred
    SPE =  np.sum(union) / np.sum(true)
    return SPE
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
    a_pos = 0
    s_pos = 0
    Dice = 0
    Precision = 0
    Recall = 0
    Jac = 0
    Spe = 0
    num= 0
    for batch_i, (img ,label,img_id) in enumerate(dataloader):
        num+=1
        img = Variable(img.float()).cuda()
        label = label.float().squeeze().detach().numpy()
        fea_map = Net(img).squeeze().cpu().detach().numpy()
        pre = np.zeros(fea_map.shape)
        # for i in range(img.shape[0]):
        #     pre[i] = Freqcam((fea_map[i]*255).astype(np.uint8),0.5*255)/255

        pre = cv2.threshold(fea_map, 0.5, 1, cv2.THRESH_BINARY)[1]
        dice = dice_coefficient(pre,label)
        jac = dice/(2-dice)
        Jac += jac
        Spe += SPE(pre,label)
        Dice+=dice
        pl = np.where(fea_map>=0.5)
        prel_pos = len(fea_map[pl])
        posl = label[pl]
        pl1 = np.where(posl==1)
        posl1 = len(posl[pl1])
        Recall += posl1/label.sum()
        Precision += (posl1+1)/(prel_pos+1)

        nl = np.where(fea_map < 0.5)
        prel_neg = len(fea_map[nl])
        negl = label[nl]
        nl1 = np.where(negl == 0)
        negl1 = len(negl[nl1])
        a_pos += (posl1+negl1)
        s_pos += (prel_pos+prel_neg)

        #print(img_id,": ",(posl1+negl1)/(prel_pos+prel_neg), dice," precision :",(posl1+1)/(prel_pos+1), "recall :",posl1/label.sum())

    print(epoch," avgacc:",a_pos/s_pos, " dice:",Dice/num ,"Jac: ",Jac/num, " SPE:",Spe/num," recall:",Recall/num, " Precision:", Precision/num)
    t = str(epoch)+" avgacc:"+str(a_pos/s_pos)+ "  dice:"+str(Dice/num) +"  Jac: ",str(Jac/num)+ " SPE:"+str(Spe/num)+" recall:"+str(Recall/num)+'\n'
   # txt = open("txt\\isic18_result3.txt",mode="a+")
   # txt.writelines(t)
if __name__ == "__main__":
   #for i in range(300):

        evaluate("E:\\pifujing\\data\\ISIC-2018\\","phi4c94")