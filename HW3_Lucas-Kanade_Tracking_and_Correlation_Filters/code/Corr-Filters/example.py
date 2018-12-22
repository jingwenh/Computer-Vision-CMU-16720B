import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage
import cv2
import os


img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])


# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5


def init():
    return [cropax, patch, all_patchax]


def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    if i < N:  # If the animation is still running
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]
    else:  # Stuff to do after the animation ends
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
                          Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

        # Place your solution code for question 4.3 here
        result_dir = '../../result/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        lam0 = 0
        g0 = np.dot(np.dot(np.linalg.inv(np.dot(X, np.transpose(X) + lam0 * np.eye(X.shape[0]))), X), Y)
        g0 = np.reshape(g0, (29, 45))
        plt.matshow(g0)
        plt.show()

        out0 = scipy.ndimage.correlate(img, g0)
        plot_image(out0, os.path.join(result_dir, 'correlate0.jpg'))

        lam1 = 1
        g1 = np.dot(np.dot(np.linalg.inv(np.dot(X, np.transpose(X)) + lam1 * np.eye(X.shape[0])), X), Y)
        g1 = np.reshape(g1, (29, 45))
        plt.matshow(g1)
        plt.show()

        out1 = scipy.ndimage.correlate(img, g1)
        plot_image(out1, os.path.join(result_dir, 'correlate1.jpg'))

        out2 = scipy.ndimage.convolve(img, g0)
        plot_image(out2, os.path.join(result_dir, 'convolve0.jpg'))
        out3 = scipy.ndimage.convolve(img, g1)
        plot_image(out3, os.path.join(result_dir, 'convolve1.jpg'))

        out4 = scipy.ndimage.convolve(img, g0[::-1, ::-1])
        plot_image(out4, os.path.join(result_dir, 'convolve2.jpg'))
        out5 = scipy.ndimage.convolve(img, g1[::-1, ::-1])
        plot_image(out5, os.path.join(result_dir, 'convolve3.jpg'))

        return []


def plot_image(image, file_path):
    normalized = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('filter response', normalized)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()
    cv2.imwrite(file_path, normalized*255)


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=10)
plt.show()

