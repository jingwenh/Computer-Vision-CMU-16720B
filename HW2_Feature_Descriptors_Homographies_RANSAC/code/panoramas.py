import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def blend_mask(im1, im2, homography1, homography2, out_shape):
    """ This is to generate the warped masks for the input images
        for further image blending.

    Args:
      im1: input image1 in numpy.array with size [H, W, 3]
      im2: input image2 in numpy.array with size [H, W, 3]
      homography1: the homography to warp image1 onto the panorama
      homography2: the homography to warp image2 onto the panorame
      out_shape: the size of the final panarama, in format of (width, height)
    Returns:
      warp_mask1: The warped mask for im1, namely the weights for im1 to blend
      warp_mask2: The warped mask for im2, namely the weights for im2 to blend
    """
    im1h, im1w, _ = im1.shape
    im2h, im2w, _ = im2.shape

    # create mask for im1, zero at the borders and 1 at the center of the image
    mask1 = np.zeros((im1h, im1w))
    mask1[0, :] = 1
    mask1[-1, :] = 1
    mask1[:, 0] = 1
    mask1[:, -1] = 1
    mask1 = distance_transform_edt(1 - mask1)
    mask1 = mask1 / np.max(mask1)
    warp_mask1 = cv2.warpPerspective(mask1, homography1, out_shape)

    # create mask for im2, zero at the borders and 1 at the center of the image
    mask2 = np.zeros((im2h, im2w))
    mask2[0, :] = 1
    mask2[-1, :] = 1
    mask2[:, 0] = 1
    mask2[:, -1] = 1
    mask2 = distance_transform_edt(1 - mask2)
    mask2 = mask2 / np.max(mask2)
    warp_mask2 = cv2.warpPerspective(mask2, homography2, out_shape)

    # combine mask1 and mask2 to calculate the weights for im1 and im2 for blending.
    sum_mask = warp_mask1 + warp_mask2
    warp_mask1 = warp_mask1 / sum_mask
    warp_mask2 = warp_mask2 / sum_mask
    warp_mask1 = np.stack((warp_mask1, warp_mask1, warp_mask1), axis=2)
    warp_mask2 = np.stack((warp_mask2, warp_mask2, warp_mask2), axis=2)

    return warp_mask1, warp_mask2


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    im1h, im1w, _ = im1.shape
    im2h, im2w, _ = im2.shape

    # project the bounding points
    bnd_points = np.array([np.array([0, 0, 1]), np.array([0, im2h-1, 1]),
                           np.array([im2w-1, 0, 1]), np.array([im2w-1, im2h-1, 1])])
    project_bnd = np.dot(H2to1, np.transpose(bnd_points))
    project_bnd = project_bnd/project_bnd[2, :]
    # get the maximum width and height
    width = int(np.round(np.max(project_bnd[0, :])))
    height = int(np.round(np.max(project_bnd[1, :])))
    # warp image2
    warp_im2 = cv2.warpPerspective(im2, H2to1, (width, height))
    cv2.imwrite('../results/6_1.jpg', warp_im2)

    # get the weights for blending
    warp_mask1, warp_mask2 = blend_mask(im1, im2, np.float32([[1,0,0],[0,1,0],[0,0,1]]), H2to1, (width, height))

    # copy image1 onto the panorama
    pano_width, pano_height = max(im1w, width), max(im1h, height)
    pano_im = np.zeros((pano_height, pano_width, 3), im1.dtype)
    pano_im[:im1h, :im1w, :] = im1

    # get the blend image with weights
    blend = pano_im * warp_mask1 + warp_im2 * warp_mask2

    # copy the warped image2 onto the panorama
    nz = tuple(np.array(np.nonzero(warp_im2)[:2]))
    pano_im[nz] = warp_im2[nz]
    # copy the overlap area onto the panorama
    ov_nz = tuple(np.array(np.nonzero(pano_im * warp_im2)[:2]))
    pano_im[ov_nz] = blend[ov_nz]

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    im1h, im1w, _ = im1.shape
    im2h, im2w, _ = im2.shape

    # project the bounding points
    bnd_points = np.array([np.array([0, 0, 1]), np.array([0, im2h - 1, 1]),
                           np.array([im2w - 1, 0, 1]), np.array([im2w - 1, im2h - 1, 1])])
    project_bnd = np.dot(H2to1, np.transpose(bnd_points))
    project_bnd = project_bnd / project_bnd[2, :]
    # get the minimum width and height, which might be negative
    min_w, min_h = int(np.round(np.min(project_bnd[0, :]))), int(np.round(np.min(project_bnd[1, :])))
    # get M for translation
    M = np.float32([[1, 0, max(-min_w, 0)], [0, 1, max(-min_h, 0)], [0, 0, 1]])

    # get the maximum width and height
    width = max(-min_w, 0) + int(np.round(np.max(project_bnd[0, :])))
    height = max(-min_h, 0) + int(np.round(np.max(project_bnd[1, :])))
    # warp image1 and image2
    warp_im1 = cv2.warpPerspective(im1, M, (width, height))
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1), (width, height))
    cv2.imwrite('../results/q6_2_warp1.jpg', warp_im1)
    cv2.imwrite('../results/q6_2_warp2.jpg', warp_im2)

    # get the weights for blending
    warp_mask1, warp_mask2 = blend_mask(im1, im2, M, np.matmul(M,H2to1), (width, height))

    # copy image1 onto the panorama
    pano_im = np.zeros((height, width, 3), im1.dtype)
    nz1 = tuple(np.array(np.nonzero(warp_im1)[:2]))
    pano_im[nz1] = warp_im1[nz1]

    # get the blend image with weights
    blend = pano_im * warp_mask1 + warp_im2 * warp_mask2

    # copy the warped image2 onto the panorama
    nz = tuple(np.array(np.nonzero(warp_im2)[:2]))
    pano_im[nz] = warp_im2[nz]
    # copy the overlap area onto the panorama
    ov_nz = tuple(np.array(np.nonzero(pano_im * warp_im2)[:2]))
    pano_im[ov_nz] = blend[ov_nz]
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)

    return pano_im


def generatePanorama(im1, im2):
    """ This is to generate the panorama given im1 and im2 by detecting and
        matching keypoints, calculating homography with RANSAC.

    Args:
      im1: input image1 in numpy.array with size [H, W, 3]
      im2: input image2 in numpy.array with size [H, W, 3]
    Returns:
      im3: stitched panorama in numpy.array.
    """
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    im3 = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_3.jpg', im3)

    return im3


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im_clip = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/panoImg_clip.png', pano_im_clip)
    cv2.imshow('panoramas', pano_im_clip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    np.save('../results/q6_1.npy', H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im3 = generatePanorama(im1, im2)