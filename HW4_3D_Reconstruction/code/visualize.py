'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
pts1, pts2 = data['pts1'], data['pts2']

M = max(im1.shape)

# estimate the fundamental matrix
F = sub.eightpoint(pts1, pts2, M)

# estimate the essential matrix E
intrinsic = np.load('../data/intrinsics.npz')
K1, K2 = intrinsic['K1'], intrinsic['K2']
E = sub.essentialMatrix(F, K1, K2)

# get the hand-selected points
coords = np.load('../data/templeCoords.npz')
x1, y1 = coords['x1'][:, 0], coords['y1'][:, 0]
SP1, SP2 = [], []
for i in range(x1.shape[0]):
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
    SP1.append([x1[i], y1[i]])
    SP2.append([x2, y2])
SP1 = np.array(SP1)
SP2 = np.array(SP2)

# get the 4 candidates of M2s
M2s = helper.camera2(E)

# calculate C1 for camera 1
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
C1 = np.dot(K1, M1)

M2 = None
P = None
# find the correct M2
for ind in range(M2s.shape[-1]):
    M2_tmp = M2s[:, :, ind]
    C2_tmp = np.dot(K2, M2_tmp)
    w, err = sub.triangulate(C1, SP1, C2_tmp, SP2)

    if np.min(w[:, -1]) > 0:
        M2 = M2_tmp
        P = w
        break

C2 = np.dot(K2, M2)

np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

# plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
