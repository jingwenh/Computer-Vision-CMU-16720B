import numpy as np
import scipy.ndimage
import skimage
import os,time

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H,W,3)
    * vgg16_weights: numpy.ndarray of shape (L,3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''
    feat = skimage.transform.resize(x, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    feat = (feat - mean) / std
    linear_count = 0
    for layer in vgg16_weights:
        if layer[0] == 'conv2d':
            feat = multichannel_conv2d(feat, layer[1], layer[2])
        elif layer[0] == 'relu':
            feat = relu(feat)
            if linear_count == 2:
                break
        elif layer[0] == 'maxpool2d':
            feat = max_pool2d(feat, layer[1])
        elif layer[0] == 'linear':
            if len(feat.shape) != 1:
                feat = np.swapaxes(feat, 0, 2)
                feat = np.swapaxes(feat, 1, 2)
            feat = feat.flatten()
            feat = linear(feat, layer[1], layer[2])
            linear_count += 1
        else:
            continue
    return feat


def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H,W,output_dim)
    '''
    h, w, d = x.shape
    output_dim = weight.shape[0]
    feat = np.zeros((h, w, output_dim))
    for i in range(output_dim):
        kernel = weight[i, :, :, :]
        out_feat = np.zeros((h, w))
        for j in range(d):
            out_feat += scipy.ndimage.convolve(x[:, :, j], kernel[j, ::-1, ::-1], mode='constant', cval=0)
        feat[:, :, i] = out_feat + bias[i]
    return feat


def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    y = np.maximum(x, 0)
    return y


def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H,W,input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size,W/size,input_dim)
    '''
    h, w, input_dim = x.shape
    y = np.zeros((h//size, w//size, input_dim))
    for i in range(h//size):
        for j in range(w//size):
            sub_x = x[i*size:(i+1)*size, j*size:(j+1)*size, :]
            y[i, j, :] = np.max(sub_x, axis=(0, 1))
    return y


def linear(x, W, b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * weight: numpy.ndarray of shape (output_dim,input_dim)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    y = np.dot(W, x) + b
    return y

