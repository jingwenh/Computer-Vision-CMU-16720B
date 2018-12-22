import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2

import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    result_dir = '../result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    cars_data = np.load('../data/carseq.npy')
    frame = cars_data[:, :, 0]
    rect_list = []
    rect = np.array([59, 116, 145, 151])
    rect_list.append(rect)
    for i in range(1, cars_data.shape[2]):
        next_frame = cars_data[:, :, i]
        p = LucasKanade.LucasKanade(frame, next_frame, rect)

        rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
        rect_list.append(rect)

        # show the image
        tmp_img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        tmp_img[:, :, 0] = next_frame
        tmp_img[:, :, 1] = next_frame
        tmp_img[:, :, 2] = next_frame
        cv2.rectangle(tmp_img, (int(round(rect[0])), int(round(rect[1]))), (int(round(rect[2])), int(round(rect[3]))), color=(0,255,255), thickness=2)
        cv2.imshow('image', tmp_img)
        cv2.waitKey(1)

        if i in [1, 100, 200, 300, 400]:
            cv2.imwrite(os.path.join(result_dir, 'q1-3_{}.jpg'.format(i)), tmp_img*255)

        frame = next_frame

    rect_list = np.array(rect_list)
    np.save(os.path.join('carseqrects.npy'), rect_list)
