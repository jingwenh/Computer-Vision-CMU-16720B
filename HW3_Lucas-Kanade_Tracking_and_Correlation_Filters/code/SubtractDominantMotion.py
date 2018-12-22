import numpy as np
import scipy.ndimage.morphology as mp
import cv2

import LucasKanadeAffine
from scipy.interpolate import RectBivariateSpline

import InverseCompositionAffine


def SubtractDominantMotion(image1, image2):
    # Input:
    # 	Images at time t and t+1
    #  Output:
    # 	mask: [nxm]
    #  put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)

    M = LucasKanadeAffine.LucasKanadeAffine(image1, image2)
    # use efficient method
    # M = InverseCompositionAffine.InverseCompositionAffine(image1, image2)

    if M.shape[0] < 3:
        M = np.vstack((M, np.array([[0, 0, 1]])))
    M = np.linalg.inv(M)

    interp_spline_image1 = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    interp_spline_image2 = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)
    # project coordinates to interpolate the values
    x = np.arange(0, image2.shape[1])
    y = np.arange(0, image2.shape[0])
    X, Y = np.meshgrid(x, y)
    X_ = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
    Y_ = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
    # get the invalid positions that are not common ares in two images
    invalid = (X_ < 0) | (X_ >= image1.shape[1]) | (Y_ < 0) & (Y_ >= image1.shape[0])
    interped_I1 = interp_spline_image1.ev(Y_, X_)
    interped_I2 = interp_spline_image2.ev(Y, X)
    interped_I1[invalid] = 0
    interped_I2[invalid] = 0

    # calculate the difference
    diff = abs(interped_I2 - interped_I1)
    th = 0.1
    ind = (diff > th) & (interped_I2 != 0)
    mask[ind] = 1
    ker = np.array(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]))
    mask = mp.binary_dilation(mask, structure=ker).astype(mask.dtype)

    return mask
