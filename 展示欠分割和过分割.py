import numpy as np
import cv2
import os

source1 = "/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/对比数据/threshold3"
source2 = "/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/对比数据/ISIC-2017_Test_v2_Part1_GroundTruth"

map = np.zeros((224,224,3))
for name in os.listdir(source1):
    if name == ".DS_Store":
        continue
    name1 = "ISIC_0"+name.replace(".jpg","_segmentation.png")
    pred = cv2.imread(source1+"/"+name,0)
    true = cv2.imread(source2+"/"+name1,0)
    print(name1)
    pred = cv2.resize(pred,(224,224))
    true = cv2.resize(true,(224,224))
    pred = cv2.threshold(pred,127,255,cv2.THRESH_BINARY)[1]
    true = cv2.threshold(true,127,255,cv2.THRESH_BINARY)[1]

    for i in range(pred.shape[0]):
        for j in range(pred.shape[0]):
            # if i<30 or j<30 :
            #     pred[i][j] = 0
            # if i>194 or j>194:
            #     pred[i][j] = 0
            if pred[i][j] == true[i][j]:
                if pred[i][j]>=127:
                  map[i][j][0],map[i][j][1],map[i][j][2] = 255,255,255
                if pred[i][j]< 127:
                  map[i][j][0],map[i][j][1],map[i][j][2] = 0, 0, 0
            if pred[i][j] > true[i][j]:
                map[i][j][0], map[i][j][1], map[i][j][2] = 0, 0, 255
            if pred[i][j] < true[i][j]:
                map[i][j][0], map[i][j][1], map[i][j][2] = 0, 255, 0

    cv2.imwrite("/Users/lixiaofan/Desktop/项目/皮肤镜1/皮肤镜/data/对比数据/threshold3_new/"+name,map)