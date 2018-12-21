"""
Check the dimensions of function arguments
This is *not* a correctness check

Written by Chen Kong, 2018.
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import submission as sub
import helper
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
helper.displayEpipolarF(im1, im2, F8)
np.savez('q2_1.npz', F=F8, M=M)

# 2.2
pts1 = np.array([[256,270],[162,152],[199,127],[147,131],[381,236],[193,290],[157,231]])
pts2 = np.array([[257,266],[161,151],[197,135],[146,133],[380,215],[194,284],[157,211]])
Farray = sub.sevenpoint(pts1, pts2, M)
helper.displayEpipolarF(im1, im2, Farray[1])
np.savez('q2_2.npz', F=Farray[1], M=M, pts1=pts1, pts2=pts2)

# 3.1
intrinsic = np.load('../data/intrinsics.npz')
K1, K2 = intrinsic['K1'], intrinsic['K2']
E = sub.essentialMatrix(F8, K1, K2)
print(E)

# 4.1
selected_pts1, selected_pts2 = helper.epipolarMatchGUI(im1, im2, F8)
#np.savez('q4_1.npz', F=F8, pts1=selected_pts1, pts2=selected_pts2)

# 5.1
noise_data = np.load('../data/some_corresp_noisy.npz')
F, inliers = sub.ransacF(noise_data['pts1'], noise_data['pts2'], M)
np.savez('tmpnew.npz', F=F, inliers=inliers)
# helper.displayEpipolarF(im1, im2, F)
# F_compare = sub.eightpoint(noise_data['pts1'], noise_data['pts2'], M)
# helper.displayEpipolarF(im1, im2, F_compare)

# 5.2
r = np.ones([3, 1])
R = sub.rodrigues(r)
assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'

R = np.eye(3)
r = sub.invRodrigues(R)
assert (r.shape == (3, )) | (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'

# 5.3
noise_data = np.load('../data/some_corresp_noisy.npz')
intrinsic = np.load('../data/intrinsics.npz')
K1, K2 = intrinsic['K1'], intrinsic['K2']
# tmp = np.load('tmp.npz')
# F, inliers = tmp['F'], tmp['inliers']
helper.displayEpipolarF(im1, im2, F)
# estimate the essential matrix E
p1, p2 = noise_data['pts1'][inliers, :], noise_data['pts2'][inliers, :]

E = np.dot(np.transpose(K2), np.dot(F, K1))
M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
C1 = np.dot(K1, M1)
M2s = helper.camera2(E)
M2_init = None
# find the correct M2
for ind in range(M2s.shape[-1]):
    M2_tmp = M2s[:, :, ind]
    C2_tmp = np.dot(K2, M2_tmp)
    w, err = sub.triangulate(C1, p1, C2_tmp, p2)

    if np.min(w[:, -1]) > 0:
        M2_init = M2_tmp

C2_init = np.dot(K2, M2_init)
P_init, err = sub.triangulate(C1, p1, C2_init, p2)
print(err)

M2, P = sub.bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c='r', marker='.')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='.')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

C2 = np.dot(K2, M2)
W = np.hstack((P, np.ones((P.shape[0], 1))))
err2 = 0
for i in range(p1.shape[0]):
    proj1 = np.dot(C1, np.transpose(W[i, :]))
    proj2 = np.dot(C2, np.transpose(W[i, :]))
    proj1 = np.transpose(proj1[:2] / proj1[-1])
    proj2 = np.transpose(proj2[:2] / proj2[-1])
    # compute error
    err2 += np.sum((proj1 - p1[i]) ** 2 + (proj2 - p2[i]) ** 2)

print(err2)
