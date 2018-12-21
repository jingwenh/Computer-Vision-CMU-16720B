import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    A = []
    for i in range(p1.shape[1]):
        A.append(np.array([p2[0, i], p2[1, i], 1, 0, 0, 0, -p1[0, i]*p2[0, i], -p1[0, i]*p2[1, i], -p1[0, i]]))
        A.append(np.array([0, 0, 0, -p2[0, i], -p2[1, i], -1, p1[1, i]*p2[0, i], p1[1, i]*p2[1, i], p1[1, i]]))

    A = np.array(A)
    u, s, vh = np.linalg.svd(A)
    H2to1 = vh[-1, :].reshape((3, 3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    p1 = []
    p2 = []
    for i in range(matches.shape[0]):
        p1.append(locs1[matches[i][0]][:2])
        p2.append(locs2[matches[i][1]][:2])
    p1 = np.array(p1)
    p2 = np.array(p2)

    max_count = 0
    bestH = np.zeros((3,3))
    while num_iter > 0:
        try:
            # randomly select four point pairs and calculate the homography
            ind = (np.random.random((1, 4)) * matches.shape[0]).astype(np.int)

            rand_p1 = np.transpose(p1[ind][0, :, :])
            rand_p2 = np.transpose(p2[ind][0, :, :])
            H2to1 = computeH(rand_p1, rand_p2)

            homo_p1 = np.transpose(np.hstack((p1, np.ones((p1.shape[0], 1)))))
            homo_p2 = np.transpose(np.hstack((p2, np.ones((p2.shape[0], 1)))))
            projected_p2 = np.dot(H2to1, homo_p2)
            projected_p2 /= projected_p2[2, :]

            dist = np.sum((homo_p1 - projected_p2) ** 2, axis=0) ** 0.5

            # compare the count of inliers
            inlier_count = dist[dist <= tol].size
            if inlier_count > max_count:
                max_count = inlier_count
                bestH = H2to1

            num_iter -= 1
        except:
            print('Bad selection of 4 points pair, can not calculate homography, skip.')

    return bestH
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

