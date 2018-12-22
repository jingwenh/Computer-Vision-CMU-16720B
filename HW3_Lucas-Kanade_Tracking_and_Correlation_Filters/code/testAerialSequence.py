import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
import cv2

import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    result_dir = '../result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    aerial_data = np.load('../data/aerialseq.npy')
    frame = aerial_data[:, :, 0]

    for i in range(1, aerial_data.shape[2]):
        next_frame = aerial_data[:, :, i]
        mask = SubtractDominantMotion.SubtractDominantMotion(frame, next_frame)

        tmp_img = np.zeros((next_frame.shape[0], next_frame.shape[1], 3))
        tmp_img[:, :, 0] = next_frame
        tmp_img[:, :, 1] = next_frame
        tmp_img[:, :, 2] = next_frame
        tmp_img[:, :, 0][mask==1] = 1
        cv2.imshow('image', tmp_img)
        cv2.waitKey(1)

        if i in [30, 60, 90, 120]:
            cv2.imwrite(os.path.join(result_dir, 'q3-3_{}.jpg'.format(i)), tmp_img*255)

        frame = next_frame
