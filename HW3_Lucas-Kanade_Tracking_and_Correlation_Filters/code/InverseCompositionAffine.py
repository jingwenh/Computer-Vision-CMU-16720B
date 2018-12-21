import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1):
    # Input:
    # 	It: template image
    # 	It1: Current image

    #  Output:
    # 	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()

    th = 0.001

    x_min, y_min, x_max, y_max = 0, 0, It.shape[1] - 1, It.shape[0] - 1
    delta_p = np.array([It1.shape[1]] * 6)

    interp_spline_It1 = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It1)
    interp_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    x = np.arange(x_min, x_max + 0.5)
    y = np.arange(y_min, y_max + 0.5)
    X, Y = np.meshgrid(x, y)

    interped_gx = interp_spline_It.ev(Y, X, dx=0, dy=1).flatten()
    interped_gy = interp_spline_It.ev(Y, X, dx=1, dy=0).flatten()

    N = interped_gx.shape[0]
    A_ = np.zeros((N, 6))
    FX, FY = X.flatten(), Y.flatten()
    A_[:, 0] = np.multiply(interped_gx, FX)
    A_[:, 1] = np.multiply(interped_gx, FY)
    A_[:, 2] = interped_gx
    A_[:, 3] = np.multiply(interped_gy, FX)
    A_[:, 4] = np.multiply(interped_gy, FY)
    A_[:, 5] = interped_gy

    while np.sum(delta_p ** 2) >= th:
        X_ = p[0] * X + p[1] * Y + p[2]
        Y_ = p[3] * X + p[4] * Y + p[5]
        valid = (X_ > 0) & (X_ < It1.shape[1]) & (Y_ > 0) & (Y_ < It1.shape[0])
        X_ = X_[valid]
        Y_ = Y_[valid]

        interped_I = interp_spline_It1.ev(Y_, X_)

        A_valid = A_[valid.flatten()]
        b = interped_I.flatten() - It[valid].flatten()
        b_ = np.dot(np.transpose(A_valid), b)

        delta_p = np.dot(np.linalg.inv(np.dot(np.transpose(A_valid), A_valid)), b_)

        M = np.vstack((np.reshape(p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M = np.vstack((np.reshape(delta_p, (2, 3)), np.array([[0, 0, 1]])))
        delta_M[0, 0] += 1
        delta_M[1, 1] += 1
        M = np.dot(M, np.linalg.inv(delta_M))

        p = M[:2, :].flatten()

    M = M[:2, :]

    return M
