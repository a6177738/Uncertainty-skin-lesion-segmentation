import numpy as np
import cv2
#
# def select(img):
#     color = list()
#     w,h = img.shape[0],img.shape[1]
#
#     gray_img = np.zeros((w,h))
#     midw,midh = int(w/2),int(h/2)
#     color.append(img[midw,midh,:])
#
#     for i in range(midw-30,midw+30):
#         for j in range(midh-30,midh+30):
#             for z in color:
#                 if img[i,j,0] != z[0] and img[i,j,1] != z[1] and img[i,j,2] != z[2]:
#                     color.append(img[i,j,:])
#     for i in range(w):
#         for j in range(h):
#             for z in color:
#                 if img[i,j,0] != z[0] and img[i,j,1] != z[1] and img[i,j,2] != z[2]:
#                     gray_img[i,j] = 0
#                 else: gray_img[i,j]= 255
#     return gray_img
# img = cv2.imread("/Users/lixiaofan/Desktop/皮肤镜1/皮肤镜/data/ISIC-2017/Ncut3/ISIC_0014768.jpg")
#
# img = select(img)
# cv2.imwrite('1.png',img)
a = -224
b = 224**2
print(b)
