import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import multiprocessing


def get_feature_for_one_image(args):
    """

    To extract the feature for image and make pair of the feature and label

    [input]
    * image_path: the path of the input image
    * label: the label of the image
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    [output]
    * feature: numpy.ndarray of shape (K)
    * label: the label of the image

    """
    image_path, label, dictionary, SPM_layer_num = args
    feature = get_image_feature(image_path, dictionary, SPM_layer_num, dictionary.shape[0])
    return feature, label


def build_recognition_system(num_workers=2):
    '''

    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    # load train data
    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    image_names = train_data['image_names']
    labels = train_data['labels']
    SPM_layer_num = 2

    image_path = [os.path.join('../data', image_name[0]) for image_name in image_names]
    args = zip(image_path, labels, [dictionary for _ in image_path], [SPM_layer_num for _ in image_path])

    # extract features for images
    pool = multiprocessing.Pool(processes=num_workers*4)
    results = pool.map(get_feature_for_one_image, args)
    features = np.array([result[0] for result in results])
    labels = np.array([result[1] for result in results])

    np.savez('trained_system.npz', features=features, labels=labels,
             dictionary=dictionary, SPM_layer_num=SPM_layer_num)


def evaluate_recognition_system(num_workers=2):
    '''

    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    # load test data
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")

    # load trained system
    train_features = trained_system['features']
    train_labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = trained_system['SPM_layer_num']

    test_image_names = test_data['image_names']
    test_image_path = [os.path.join('../data', item[0]) for item in test_image_names]
    test_labels = test_data['labels']

    # obtain the feature for test image
    pool = multiprocessing.Pool(processes=num_workers*4)
    args = zip(test_image_path, test_labels, [dictionary for _ in test_image_path],
               [SPM_layer_num for _ in test_image_path])
    test_result = pool.map(get_feature_for_one_image, args)
    test_features = [result[0] for result in test_result]
    test_labels = [result[1] for result in test_result]

    # calculate the confusion matrix
    class_num = max(len(set(test_labels)), len(set(train_labels)))
    conf = np.zeros((class_num, class_num))
    for i, feature in enumerate(test_features):
        sim = distance_to_set(feature, train_features)
        [index] = np.where(sim == np.max(sim))[0]
        predict_label = train_labels[index]
        true_label = test_labels[i]
        conf[true_label, predict_label] += 1

    accuracy = np.diag(conf).sum()/conf.sum()

    return conf, accuracy


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = skimage.io.imread(file_path)
    image = image.astype('float') / 255
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    return feature


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    intersection = np.minimum(word_hist, histograms)
    sim = np.sum(intersection, axis=1)
    return sim


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K
    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist, bin_edges = np.histogram(wordmap.flatten(), bins=range(dict_size+1), density=True)
    hist = hist / np.sum(hist)
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    h, w = wordmap.shape
    cnum = int(2**layer_num)
    cell_h, cell_w = int(h//cnum), int(w//cnum)

    hist_all = np.empty((0,), np.float)

    # calculate the hist for each minimum cell
    weight = 1/2
    cell_hist_arr = np.zeros((cnum, cnum, dict_size))
    for cell_ind in range(cnum*cnum):
        row_ind = int(cell_ind//cnum)
        col_ind = cell_ind % cnum
        cell = wordmap[row_ind*cell_h:(row_ind+1)*cell_h, col_ind*cell_w:(col_ind+1)*cell_w]
        cell_hist = get_feature_from_wordmap(cell, dict_size)
        cell_hist_arr[row_ind, col_ind, :] = cell_hist
    hist_all = np.append((cell_hist_arr*weight).flatten(), hist_all)

    # for each layer, add corresponding cells from the previous layer
    pre_layer_hist = np.copy(cell_hist_arr)
    for l in range(layer_num-1, -1, -1):
        cnum //= 2
        weight /= (2 if l != 0 else 1)
        layer_hist = np.zeros((cnum, cnum, dict_size))
        for ind in range(cnum*cnum):
            row_ind = int(ind//cnum)
            col_ind = int(ind%cnum)
            layer_hist[row_ind, col_ind, :] = np.sum(pre_layer_hist[row_ind*2:(row_ind+1)*2,
                                                     col_ind*2:(col_ind+1)*2, :], axis=(0, 1))

        hist_all = np.append((layer_hist*weight).flatten(), hist_all)
        pre_layer_hist = layer_hist

    # normalize the hist
    hist_all = hist_all/np.sum(hist_all)

    return hist_all
