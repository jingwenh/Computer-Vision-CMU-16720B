"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import helper
import matplotlib.pyplot as plt
import scipy.optimize


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # normalize the coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    x1, y1, x2, y2 = x1/M, y1/M, x2/M, y2/M
    # normalization matrix
    T = np.array([[1./M, 0, 0], [0, 1./M, 0], [0, 0, 1]])

    A = np.transpose(np.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, np.ones(x1.shape))))

    # get F by SVD decomposition
    u, s, vh = np.linalg.svd(A)
    f = vh[-1, :]
    F = np.reshape(f, (3, 3))

    # refine F
    F = helper.refineF(F, pts1/M, pts2/M)

    # constraint of rank 2 by setting the last singular value to 0
    F = helper._singularize(F)

    # rescale the data
    F = np.dot(np.transpose(T), np.dot(F, T))

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # normalize the coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    x1, y1, x2, y2 = x1 / M, y1 / M, x2 / M, y2 / M
    # normalization matrix
    T = np.array([[1. / M, 0, 0], [0, 1. / M, 0], [0, 0, 1]])

    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))))

    # get F by SVD decomposition
    u, s, vh = np.linalg.svd(A)
    f1 = vh[-1, :]
    f2 = vh[-2, :]
    F1 = np.reshape(f1, (3, 3))
    F2 = np.reshape(f2, (3, 3))

    fun = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    # get the coefficients of the polynomial
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3 - (fun(2)-fun(-2))/12
    a2 = (fun(1)+fun(-1))/2 - a0
    a3 = (fun(1)-fun(-1))/2 - a1
    # solve for alpha
    alpha = np.roots([a3, a2, a1, a0])

    Farray = [a*F1+(1-a)*F2 for a in alpha]
    # refine F
    Farray = [helper.refineF(F, pts1/M, pts2/M) for F in Farray]
    # denormalize F
    Farray = [np.dot(np.transpose(T), np.dot(F, T)) for F in Farray]

    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.dot(np.transpose(K2), np.dot(F, K1))
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    A1 = np.vstack((C1[0, 0]-C1[2, 0]*x1, C1[0, 1]-C1[2, 1]*x1, C1[0, 2]-C1[2, 2]*x1, C1[0, 3]-C1[2, 3]*x1)).transpose()
    A2 = np.vstack((C1[1, 0]-C1[2, 0]*y1, C1[1, 1]-C1[2, 1]*y1, C1[1, 2]-C1[2, 2]*y1, C1[1, 3]-C1[2, 3]*y1)).transpose()
    A3 = np.vstack((C2[0, 0]-C2[2, 0]*x2, C2[0, 1]-C2[2, 1]*x2, C2[0, 2]-C2[2, 2]*x2, C2[0, 3]-C2[2, 3]*x2)).transpose()
    A4 = np.vstack((C2[1, 0]-C2[2, 0]*y2, C2[1, 1]-C2[2, 1]*y2, C2[1, 2]-C2[2, 2]*y2, C2[1, 3]-C2[2, 3]*y2)).transpose()

    # calculate the 3D coordinates for each point
    N = pts1.shape[0]
    w = np.zeros((N, 3))
    for ind in range(N):
        A = np.vstack((A1[ind, :], A2[ind, :], A3[ind, :], A4[ind, :]))
        u, s, vh = np.linalg.svd(A)
        p = vh[-1, :]
        w[ind, :] = p[:3]/p[-1]

    # project to 2D points
    W = np.hstack((w, np.ones((N, 1))))
    err = 0
    for i in range(N):
        proj1 = np.dot(C1, np.transpose(W[i, :]))
        proj2 = np.dot(C2, np.transpose(W[i, :]))
        proj1 = np.transpose(proj1[:2]/proj1[-1])
        proj2 = np.transpose(proj2[:2]/proj2[-1])
        # compute error
        err += np.sum((proj1-pts1[i])**2 + (proj2-pts2[i])**2)

    return w, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # set the size of the window
    x1, y1 = int(round(x1)), int(round(y1))
    window_size = 11
    center = window_size//2
    sigma = 5
    search_range = 40

    # create gaussian weight matrix
    mask = np.ones((window_size, window_size))*center
    mask = np.repeat(np.array([range(window_size)]), window_size, axis=0) - mask
    mask = np.sqrt(mask**2+np.transpose(mask)**2)
    weight = np.exp(-0.5*(mask**2)/(sigma**2))
    weight /= np.sum(weight)

    if len(im1.shape) > 2:
        weight = np.repeat(np.expand_dims(weight, axis=2), im1.shape[-1], axis=2)

    # get the epipolar line
    p = np.array([[x1], [y1], [1]])
    l2 = np.dot(F, p)

    # get the patch around the pixel in image1
    patch1 = im1[y1-center:y1+center+1, x1-center:x1+center+1]
    # get the points on the epipolar line
    h, w, _ = im2.shape
    Y = np.array(range(y1-search_range, y1+search_range))
    X = np.round(-(l2[1]*Y+l2[2])/l2[0]).astype(np.int)
    valid = (X >= center) & (X < w - center) & (Y >= center) & (Y < h - center)
    X, Y = X[valid], Y[valid]

    min_dist = None
    x2, y2 = None, None
    for i in range(len(X)):
        # get the patch around the pixel in image2
        patch2 = im2[Y[i]-center:Y[i]+center+1, X[i]-center:X[i]+center+1]
        # calculate the distance
        dist = np.sum((patch1-patch2)**2*weight)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            x2, y2 = X[i], Y[i]

    return x2, y2

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    N = pts1.shape[0]
    iter = 1000
    thresh = 1
    max_inlier = 0
    F = None
    inliers = None

    for i in range(iter):
        inds = np.random.randint(0, N, (7,))
        F7s = sevenpoint(pts1[inds, :], pts2[inds, :], M)

        for F7 in F7s:
            # calculate the epipolar lines
            pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, N))))
            l2s = np.dot(F7, pts1_homo)
            l2s = l2s/np.sqrt(np.sum(l2s[:2, :]**2, axis=0))

            # calculate the deviation of pts2 away from the epiploar lines
            pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, N))))
            deviate = abs(np.sum(pts2_homo*l2s, axis=0))

            # determine the inliners
            tmp_inliers = np.transpose(deviate < thresh)

            if tmp_inliers[tmp_inliers].shape[0] > max_inlier:
                max_inlier = tmp_inliers[tmp_inliers].shape[0]
                F = F7
                inliers = tmp_inliers

    print(max_inlier/N)

    return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.sqrt(np.sum(r**2))
    n = r/theta if theta != 0 else r
    n_cross = np.array([[0, -n[2, 0], n[1, 0]], [n[2, 0], 0, -n[0, 0]], [-n[1, 0], n[0, 0], 0]])
    I = np.eye(3)
    n_cross_square = np.dot(n, np.transpose(n)) - np.sum(n**2)*I
    R = I + np.sin(theta)*n_cross + (1-np.cos(theta))*n_cross_square

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    A = (R - np.transpose(R))/2
    p = np.array([[A[2, 1]], [A[0, 2]], [A[1, 0]]])
    s = np.sqrt(np.sum(p**2))
    c = (R[0, 0]+R[1, 1]+R[2, 2]-1)/2

    if s == 0. and c == 1.:
        r = np.zeros((3, 1))
        return r

    elif s == 0. and c == -1.:
        tmp = R + np.diag(np.array([1, 1, 1]))
        v = None
        for i in range(3):
            if np.sum(tmp[:, i]) != 0:
                v = tmp[:, i]
                break
        u = v/np.sqrt(np.sum(v**2))
        r = np.reshape(u*np.pi, (3, 1))
        if np.sqrt(np.sum(r**2)) == np.pi and ((r[0, 0] == 0. and r[1, 0] == 0. and r[2, 0] < 0)
                                               or (r[0, 0] == 0. and r[1, 0] < 0) or (r[0, 0] < 0)):
            r = -r
        return r
    else:
        u = p / s
        theta = np.arctan2(np.float(s), np.float(c))
        r = u*theta
        return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P, r2, t2 = x[:-6], x[-6:-3], x[-3:]

    N = P.shape[0]//3
    P = np.reshape(P, (N, 3))
    r2 = np.reshape(r2, (3, 1))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    P = np.vstack((np.transpose(P), np.ones((1, N))))
    p1_hat = np.dot(np.dot(K1, M1), P)
    p1_hat = np.transpose(p1_hat[:2, :]/p1_hat[2, :])
    p2_hat = np.dot(np.dot(K2, M2), P)
    p2_hat = np.transpose(p2_hat[:2, :]/p2_hat[2, :])

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]), (p2-p2_hat).reshape([-1])])

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    R2_init, t2_init = M2_init[:, :3], M2_init[:, 3]
    r2_init = invRodrigues(R2_init).reshape([-1])
    x = np.concatenate([P_init.reshape([-1]), r2_init, t2_init])

    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    x_update = scipy.optimize.minimize(func, x).x

    P, r2, t2 = x_update[:-6], x_update[-6:-3], x_update[-3:]

    N = P.shape[0] // 3
    P2 = np.reshape(P, (N, 3))

    r2 = np.reshape(r2, (3, 1))
    t2 = np.reshape(t2, (3, 1))
    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2))

    return M2, P2
