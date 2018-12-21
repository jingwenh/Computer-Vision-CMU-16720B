import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeBasis(It, It1, rect, bases):
    # Input:
    # 	It: template image
    # 	It1: Current image
    # 	rect: Current position of the car
    # 	(top left, bot right coordinates)
    # 	bases: [n, m, k] where nxm is the size of the template.
    # Output:
    # 	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    th = 0.0001
    B = []
    for i in range(bases.shape[2]):
        B.append(bases[:, :, i].flatten())
    B = np.transpose(np.array(B))
    E = np.diag([1]*B.shape[0])

    B_null = E - np.dot(B, np.transpose(B))

    x_min, y_min, x_max, y_max = rect[0], rect[1], rect[2], rect[3]
    delta_p = np.array([It.shape[1], It.shape[0]])
    p = np.zeros(2)

    interp_spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interp_spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    while np.sum(delta_p ** 2) >= th:
        x_ = np.arange(x_min + p[0], x_max + 0.5 + p[0])
        y_ = np.arange(y_min + p[1], y_max + 0.5 + p[1])
        X_, Y_ = np.meshgrid(x_, y_)
        interped_I = interp_spline_It1.ev(Y_, X_)

        x = np.arange(x_min, x_max + 0.1)
        y = np.arange(y_min, y_max + 0.1)
        X, Y = np.meshgrid(x, y)
        interped_It = interp_spline_It.ev(Y, X)

        # calculate gradients
        interped_gx = interp_spline_It1.ev(Y_, X_, dx=0, dy=1).flatten()
        interped_gy = interp_spline_It1.ev(Y_, X_, dx=1, dy=0).flatten()
        N = interped_gx.shape[0]

        # get matrix A
        A = np.zeros((N, 2))
        A[:, 0] = interped_gx
        A[:, 1] = interped_gy

        # apply bases matrix
        A_ = np.dot(B_null, A)

        # get matrix b
        b = interped_It.flatten() - interped_I.flatten()
        b = np.expand_dims(b, axis=1)
        # apply bases matrix
        b_ = np.dot(B_null, b)

        delta_p = np.dot(np.linalg.inv(np.dot(np.transpose(A_), A_)), np.dot(np.transpose(A_), b_))

        # update p
        p += delta_p.flatten()

    return p

