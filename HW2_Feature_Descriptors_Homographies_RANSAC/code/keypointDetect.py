import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    # compute DoG_pyramid here
    for i in range(len(levels)-1):
        DoG_pyramid.append(gaussian_pyramid[:, :, i+1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)

    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = np.zeros(DoG_pyramid.shape)
    # Compute principal curvature here
    for l in range(DoG_pyramid.shape[-1]):
        Dx = cv2.Sobel(DoG_pyramid[:, :, l], ddepth=-1, dx=1, dy=0)
        Dy = cv2.Sobel(DoG_pyramid[:, :, l], ddepth=-1, dx=0, dy=1)
        Dxx = cv2.Sobel(Dx, ddepth=-1, dx=1, dy=0)
        Dxy = cv2.Sobel(Dx, ddepth=-1, dx=0, dy=1)
        Dyx = cv2.Sobel(Dy, ddepth=-1, dx=1, dy=0)
        Dyy = cv2.Sobel(Dy, ddepth=-1, dx=0, dy=1)

        det = Dxx*Dyy - Dxy*Dyx
        det[det==0.] = 10**(-20)
        principal_curvature[:, :, l] = ((Dxx + Dyy)**2)/det

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    # Compute locsDoG here
    imH, imW, l = DoG_pyramid.shape
    for i in range(0, l):
        # find neighbors
        neighbors = [DoG_pyramid[:imH-2, :imW-2, i], DoG_pyramid[:imH-2, 1:imW-1, i],
                     DoG_pyramid[:imH-2, 2:, i], DoG_pyramid[1:imH-1, :imW-2, i],
                     DoG_pyramid[1:imH-1, 2:, i], DoG_pyramid[2:, :imW-2, i],
                     DoG_pyramid[2:, 1:imW-1, i], DoG_pyramid[2:, 2:, i]]
        if i > 0:
            neighbors.append(DoG_pyramid[1:imH-1, 1:imW-1, i-1])
        if i < l-1:
            neighbors.append(DoG_pyramid[1:imH-1, 1:imW-1, i+1])
        neighbors = np.array(neighbors)
        layer_data = DoG_pyramid[1:imH-1, 1:imW-1, i]
        # find local extrema
        is_extrema = (layer_data > np.max(neighbors, axis=0)) | (layer_data < np.min(neighbors, axis=0))
        # apply threshold to exclude unqualified points
        is_extrema &= (np.absolute(layer_data) > th_contrast)
        is_extrema &= (np.absolute(principal_curvature[1:imH-1, 1:imW-1, i]) < th_r)
        spatial_loc = np.where(is_extrema == True)

        y, x = spatial_loc[0]+1, spatial_loc[1]+1
        for ind in range(len(y)):
            locsDoG.append([x[ind], y[ind], DoG_levels[i]])

    locsDoG = np.array(locsDoG)

    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels=levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                              th_contrast=th_contrast, th_r=th_r)

    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    tmp_im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
    for point in list(locsDoG):
        cv2.circle(tmp_im, (2*point[0], 2*point[1]), 2, (0, 255, 0), -1)
    cv2.imwrite('../detected_keypoints.jpg', tmp_im)


