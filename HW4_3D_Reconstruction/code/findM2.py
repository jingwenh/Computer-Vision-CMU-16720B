'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
pts1, pts2 = data['pts1'], data['pts2']

# get the scalar
M = max(im1.shape)

# estimate the fundamental matrix F
F = sub.eightpoint(pts1, pts2, M)

# estimate the essential matrix E
intrinsic = np.load('../data/intrinsics.npz')
K1, K2 = intrinsic['K1'], intrinsic['K2']
E = sub.essentialMatrix(F, K1, K2)

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
    w, err = sub.triangulate(C1, pts1, C2_tmp, pts2)

    if np.min(w[:, -1]) > 0:
        M2 = M2_tmp
        P = w
        break

C2 = np.dot(K2, M2)

np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
