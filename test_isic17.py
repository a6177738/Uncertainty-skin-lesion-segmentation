import torch
import numpy as np
import os
import cv2


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
np.set_printoptions(100000000)
def evalua(epoch):

    np.set_printoptions(threshold=10000)
    a_pos = 0
    s_pos = 0
    Dice = 0
    Precision = 0
    Recall = 0
    Jac = 0
    Spe = 0
    num= 0
    for pre_name in os.listdir("data/ISIC-2017/result"):
        if pre_name == '.DS_Store':
            continue

        fea_map = cv2.imread("data/ISIC-2017/result/"+pre_name,0)
        truth_name = pre_name.replace(".jpg","_segmentation.png").strip()
        label = cv2.imread("data/ISIC-2017/ISIC-2017_Test_v2_Part1_GroundTruth/"+truth_name,0)

        fea_map = cv2.resize(fea_map,(224,224))
        label = cv2.resize(label,(224,224))/255

        #pre = cv2.threshold(fea_map, 140, 255, cv2.THRESH_BINARY)[1]

        fea_map = cv2.threshold(fea_map, 0.532*255, 255, cv2.THRESH_BINARY)[1]


        num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(fea_map, connectivity=8)

        max = 0
        n = 500
        if num_objects > 1:
            for i in range(0, num_objects):
                loc = np.where(labels == i)
                if fea_map[loc].mean()>1 and  stats[i][4]>300:
                    if n>abs(loc[0].mean()-112)+abs(loc[1].mean()-112):
                        n = abs(loc[0].mean()-112)+abs(loc[1].mean()-112)
                        max = i
            for i in range(0, num_objects):
                if i != max:
                    loc = np.where(labels == i)
                    fea_map[loc] = 0
        pre = fea_map/255
        label = cv2.threshold(label,0.5, 1, cv2.THRESH_BINARY)[1]
        dice = dice_coefficient(pre,label)
        jac = dice/(2-dice)
        Jac += jac
        spe = SPE(pre,label)
        Spe += spe
        Dice+=dice
        pl = np.where(fea_map>=0.532*255)
        prel_pos = len(fea_map[pl])
        posl = label[pl]
        pl1 = np.where(posl==1)
        posl1 = len(posl[pl1])
        Recall += posl1/label.sum()
        Precision += (posl1+1)/(prel_pos+1)

        nl = np.where(fea_map < 0.532*255)
        prel_neg = len(fea_map[nl])
        negl = label[nl]
        nl1 = np.where(negl == 0)
        negl1 = len(negl[nl1])
        a_pos += (posl1+negl1)
        s_pos += (prel_pos+prel_neg)
        num+=1

        #print(pre_name,": Acc",(posl1+negl1)/(prel_pos+prel_neg)," Dice:", dice," precision :",(posl1+1)/(prel_pos+1), "recall :",posl1/label.sum(), " SPE:", spe, " JAC:",jac)

    print("epoch" ,epoch, " avgacc:",a_pos/s_pos, " dice:",Dice/num, " Jac: ",Jac/num, " SPE:",Spe/num," recall:",Recall/num, " precision:", Precision/num)
if __name__ == "__main__":
        evalua(1)