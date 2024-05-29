import cv2
import numpy as np
def Freqcam(activation_map, T):
    region = cv2.threshold(activation_map, T, 255, cv2.THRESH_BINARY)[1]

    region1 = region.copy()
    num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(region1, connectivity=8)

    max = 0
    second = 0
    if num_objects > 1:
        for i in range(0, num_objects):
            if stats[i][4] > max:
                max = stats[i][4]
        for i in range(0, num_objects):
            if stats[i][4] > second and stats[i][4] != max:
                second = stats[i][4]
        for i in range(0, num_objects):
            if stats[i][4] == max:
                loc = np.where(labels == i)
                if activation_map[loc].mean() < 100:
                    max = second
            if stats[i][4] < max:
                loc = np.where(labels == i)
                activation_map[loc] = 0
    return activation_map