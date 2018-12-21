import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # get the image shape
    image_shape = image.shape
    h, w = image_shape[0], image_shape[1]
    # convert the image into 3 channel if it is not
    if len(image_shape) == 2:
        image = np.matlib.repmat(image, 3, 1)
    else:
        if image_shape[-1] == 4:
            image = image[:, :, :-1]

    # convert the image into lab color space
    lab_image = skimage.color.rgb2lab(image)

    # define the filter scales
    filter_scales = [1, 2, 4, 8, 8*(2**0.5)]
    filter_responses = np.zeros((h, w, 3*4*len(filter_scales)))
    for ind, scale in enumerate(filter_scales):
        # do filter for each channel of the image
        for i in range(3):
            # Gaussian filter
            response1 = scipy.ndimage.gaussian_filter(lab_image[:, :, i], scale)
            # Laplacian of Gaussian filter
            response2 = scipy.ndimage.gaussian_laplace(lab_image[:, :, i], scale)
            # derivative of Gaussian in the x direction
            response3 = scipy.ndimage.gaussian_filter(lab_image[:, :, i], scale, [0, 1])
            # derivative of Gaussian in the y direction
            response4 = scipy.ndimage.gaussian_filter(lab_image[:, :, i], scale, [1, 0])

            filter_responses[:, :, ind*12+i] = response1
            filter_responses[:, :, ind*12+3+i] = response2
            filter_responses[:, :, ind*12+6+i] = response3
            filter_responses[:, :, ind*12+9+i] = response4

    return filter_responses


def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(image)
    h, w, _ = filter_response.shape
    wordmap = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            pixel_response = filter_response[i, j, :]
            dist = scipy.spatial.distance.cdist(np.array([pixel_response]), dictionary, 'euclidean')
            [best_match] = np.where(dist == np.min(dist))[1]
            wordmap[i, j] = best_match

    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    i, alpha, image_path = args

    image = skimage.io.imread(image_path)
    image = image.astype('float') / 255
    filter_response = extract_filter_responses(image)
    h, w, c = filter_response.shape

    # randomly choose the index for pixels and get the response
    pixel_ind = np.random.permutation(h*w)[:alpha]
    sampled_response = np.zeros((alpha, c))
    for j in range(alpha):
        h_ind = int(pixel_ind[j] // w)
        w_ind = pixel_ind[j] % w
        sampled_response[j, :] = filter_response[h_ind, w_ind, :]
    tmp_dir = '../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    np.save(os.path.join(tmp_dir, str(i)+'.npy'), sampled_response)


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    train_data = np.load("../data/train_data.npz")
    image_names = train_data['image_names']
    labels = train_data['labels']
    k = 150
    alpha = 50
    image_paths = [os.path.join('../data', item[0]) for item in image_names]

    pool = multiprocessing.Pool(processes=num_workers*4)
    args = zip(range(len(image_paths)), [alpha for _ in image_paths], image_paths)
    pool.map(compute_dictionary_one_image, args)

    # collect all the responses
    filter_responses = np.array([])
    tmp_dir = '../tmp_50'
    for file in os.listdir(tmp_dir):
        sampled_response = np.load(os.path.join(tmp_dir, file))
        filter_responses = np.array(np.append(filter_responses, sampled_response, axis=0)
                                    if filter_responses.shape[0] != 0 else sampled_response)

    # cluster the responses to generate the dictionary
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(filter_responses)
    dictionary = kmeans.cluster_centers_

    np.save('dictionary.npy', dictionary)


