import os
import cv2
import numpy as np
def dice_coefficient(y_pred, y_true):
    """dice coefficient 2nt/na + nb."""
    y_true = y_true / 255
    y_pred = y_pred / 255
    union = y_true * y_pred
    dice = 2 * np.sum(union) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def evalua(source_path,epoch,S):
    n_dice = 0
    n = 0
    for truth_name in os.listdir(
            source_path+"ISIC-2017_Test_v2_Part1_GroundTruth\\"):
        if truth_name == '.DS_Store':
            continue
        ground_truth = cv2.imread(source_path+"ISIC-2017_Test_v2_Part1_GroundTruth\\"+truth_name,0)
        ground_truth = cv2.resize(ground_truth, (512, 512))
        predict = cv2.imread(source_path+"result\\" + truth_name, 0)
        dice = dice_coefficient(predict, ground_truth)
        n_dice += dice
        n += 1
    d = n_dice / n
    print(epoch, ": ", d,"  threshold:", S)

if __name__ == '__main__':

    evalua()