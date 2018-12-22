import cv2
import numpy as np

from planarH import computeH

import matplotlib.pyplot as plt


def compute_extrinsics(K, H):
    """ This is to compute the extrinsic rotation R and translation t
        given the camera intrinsic parameters K and the estimates
        homography H.

    Args:
      K: The intrinsic matrix, with size (3*3)
      H: The homography projecting the 3D points onto the image
    Returns:
      R: The extrinsic rotation matrix with size (3*3)
      t: The extrinsic translation matrix with size (3*1)
    """
    Kinv = np.linalg.inv(K)
    H_ = np.dot(Kinv, H)
    u, s, vh = np.linalg.svd(H_[:, :2])
    R = np.dot(np.dot(u, np.array([[1, 0], [0, 1], [0, 0]])), vh)
    last_column = np.cross(R[:, 0], R[:, 1], axis=0)
    last_column /= np.sum(last_column**2)

    R = np.hstack((R, np.transpose([last_column])))
    if np.linalg.det(R) == -1:
        R[:, 2] *= -1

    t = H_[:, 2]
    scale = np.sum(H_[:, :2]/R[:, :2])/6
    t /= scale
    t = np.transpose([t])

    return R, t


def project_extrinsics(K, W, R, t):
    """ This is to project the 3D points in homogeneous coordinates
        onto the 2D image given the intrinsic matrix K and extrinsic
        matrix R and t

      K: The intrinsic matrix with size (3*3)
      W: The matrix containing 3D points in homogeneous coordinates, with
         size (N*4), N is the number of the points
      R: The extrinsic rotation matrix with size (3*3)
      t: The extrinsic translation matrix with size (3*1)
    Returns:
      X: The matrix containing the projected 2D points on the image in
         homogeneous coordinates, with size (N*3), N is the number of points.
    """
    ttt = np.dot(K, np.hstack((R, t)))
    X = np.dot(ttt, W)
    X = X/X[-1, :]

    return X


def calc_H():
    """ This is to calculate the homography projecting the 3D points onto
        2D image space, given the coordinates of 4 pairs of points both in
        3D space and 2D space.

    Returns:
      H: The homography matrix with size (3*3)
    """
    p1 = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    p2 = np.array([[0.0, 18.2, 18.2, 0.0], [0.0, 0.0, 26.0, 26.0]])
    H = computeH(p1, p2)

    return H


def get_target_point(im):
    """ This is to get the target point selected by users with mouse click.
        With the selected target point on the given image im, the object in
        the 3D space would be projected onto that point.

    Args:
      im: The input image in type of numpy array
    Returns:
      target: The selected point in homogeneous coordinates with size (3*1)
    """
    mouse_params = {'target': None, 'current_pos': None}
    title = 'get_target'
    cv2.namedWindow(title, 0)
    # resize the image to shown in a reasonable size
    scale = 400./im.shape[1]
    resize_im = cv2.resize(im, (400, int(im.shape[0]*scale)))

    def onMouse(event, x, y, flags, param):
        x = int(x/scale)
        y = int(y/scale)
        param['current_pos'] = (x, y)

        if flags & cv2.EVENT_FLAG_LBUTTON:
            param['target'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, resize_im)

    while mouse_params['target'] is None:
        im_draw = np.copy(resize_im)

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    cv2.destroyWindow(title)

    # get the homogeneous coordinates of the target point
    target = np.transpose(np.array([[mouse_params['target'][0], mouse_params['target'][1], 1.0]]))

    return target


def project(K, W, R, t, target):
    """ This is to project the points in 3D space onto the image in 2D space
        given the intrinsic matrix K and extrinsic matrix R and t, and the
        target point where want to place the 3D object to

    Args:
      K: The intrinsic matrix with size (3*3)
      W: The matrix containing 3D points in homogeneous coordinates, with
         size (N*4), N is the number of the points
      R: The extrinsic rotation matrix with size (3*3)
      t: The extrinsic translation matrix with size (3*1)
      target: The selected point in homogeneous coordinates with size (3*1)
    Returns:
      X: The matrix containing the projected 2D points on the image in
         homogeneous coordinates, with size (N*3), N is the number of points.
    """
    # Project the 3D points onto the image in 2D space
    X = project_extrinsics(K, W, R, t)
    # Find the bottom point of the object in 3D space and get its projected
    # coordinates
    z = W[2, :]
    bottom_projected = np.transpose([X[:, np.where(z == np.min(z))[0][0]]])

    # Get the translation matrix to move to the target point
    shift = target-bottom_projected
    shift[2] = 1
    T = np.hstack((np.array([[1, 0], [0, 1], [0, 0]]), shift))

    # recalculate the projected coordinates
    X = np.dot(T, X)
    X /= X[-1, :]

    return X


if __name__ == '__main__':
    # Get the extrinsic matrix
    H = calc_H()
    K = np.array([[3043.72, 0.0, 1196.00], [0.0, 3043.72, 1604.00], [0.0, 0.0, 1.0]])
    R, t = compute_extrinsics(K, H)

    # Get the matrix of points in 3D space, present in homogeneous coordinates
    W = []
    for line in open('../data/sphere.txt', 'r').readlines():
        coord = [c for c in line.strip('\n').split(' ') if c != '']
        coord = list(map(eval, coord))
        W.append(coord)
    W = np.vstack((W, np.ones((1, len(W[0])))))

    im = cv2.imread('../data/prince_book.jpeg')

    # the original projection
    X_ = project_extrinsics(K, W, R, t)
    fig_X_ = plt.figure('original_projection')
    plt.imshow(im)
    plt.plot(X_[0, :], X_[1, :], 'y.', linewidth=1, markersize=1)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig_X_)

    # select the target point
    target = get_target_point(im)

    # Project the points onto the image in 2D space, with the selected target point
    X = project(K, W, R, t, target)
    fig = plt.figure('projection_with_target_point')
    plt.imshow(im)
    plt.plot(X[0, :], X[1, :], 'y.', linewidth=1, markersize=1)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)
