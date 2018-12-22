import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the car
    #   (top left, bot right coordinates)
    #   p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #   p: movement vector [dp_x, dp_y]

    #  Put your implementation here
    th = 0.005

    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]
    delta_p = np.array([It.shape[1], It.shape[0]])
    p = p0

    interp_spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interp_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    while np.sum(delta_p**2) >= th:
        x_ = np.arange(x_min+p[0], x_max+0.1+p[0])
        y_ = np.arange(y_min+p[1], y_max+0.1+p[1])
        X_, Y_ = np.meshgrid(x_, y_)
        interped_I = interp_spline_It1.ev(Y_, X_)

        x = np.arange(x_min, x_max + 0.1)
        y = np.arange(y_min, y_max + 0.1)
        X, Y = np.meshgrid(x, y)

        # get the warped interpolated values
        interped_It = interp_spline_It.ev(Y, X)

        # calculate gradient
        interped_gx = interp_spline_It1.ev(Y_, X_, dx=0, dy=1).flatten()
        interped_gy = interp_spline_It1.ev(Y_, X_, dx=1, dy=0).flatten()

        N = interped_gx.shape[0]
        # get matrix A
        A = np.zeros((N, 2))
        A[:, 0] = interped_gx
        A[:, 1] = interped_gy
        # get matrix b
        b = interped_It.flatten() - interped_I.flatten()

        delta_p = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), b))

        # update p
        p += delta_p.flatten()

    return p

