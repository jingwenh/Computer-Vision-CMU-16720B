import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy.ndimage
import shutil

def build_recognition_system(vgg16, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''
	train_data = np.load("../data/train_data.npz")
	# load train data
	image_names = train_data['image_names']
	labels = train_data['labels']
	image_path = [os.path.join('../data', item[0]) for item in image_names]

	# get features of the training images
	pool = multiprocessing.Pool(processes=num_workers*4)
	args = zip(range(len(image_path)), image_path, [vgg16 for _ in image_path])
	pool.map(get_image_feature, args)

	features = [None]*len(image_path)
	tmp_dir = '../vgg_tmp'
	for file in os.listdir(tmp_dir):
		feature = np.load(os.path.join(tmp_dir, file))
		index = int(file.split('.')[0])
		features[index] = feature
	features = np.array(features)

	np.savez('trained_system_deep.npz', features=features, labels=labels)


def evaluate_recognition_system(vgg16, num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''
	tmp_dir = '../vgg_tmp'
	if os.path.exists(tmp_dir):
		# remove if exists
		shutil.rmtree(tmp_dir)

	# load test data
	test_data = np.load("../data/test_data.npz")
	test_image_names = test_data['image_names']
	test_image_path = [os.path.join('../data', item[0]) for item in test_image_names]
	test_labels = test_data['labels']

	# load trained system
	trained_system_deep = np.load("trained_system_deep.npz")
	train_features = trained_system_deep['features']
	train_labels = trained_system_deep['labels']

	# extract features for test data
	pool = multiprocessing.Pool(processes=num_workers*4)
	args = zip(range(len(test_image_path)), test_image_path, [vgg16 for _ in test_image_path])
	pool.map(get_image_feature, args)

	test_features = [None] * len(test_image_path)
	for file in os.listdir(tmp_dir):
		feature = np.load(os.path.join(tmp_dir, file))
		index = int(file.split('.')[0])
		test_features[index] = feature

	# calculate the confusion matrix
	class_num = max(len(set(test_labels)), len(set(train_labels)))
	conf = np.zeros((class_num, class_num))
	for i, feature in enumerate(test_features):
		dist = distance_to_set(feature, train_features)
		[index] = np.where(dist == np.max(dist))[0]
		predict_label = train_labels[index]
		true_label = test_labels[i]
		conf[true_label, predict_label] += 1

	accuracy = np.diag(conf).sum() / conf.sum()

	return conf, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	image_shape = image.shape
	# convert the image into 3 channel if it is not
	if len(image_shape) == 2:
		image = np.matlib.repmat(image, 3, 1)
	else:
		if image_shape[-1] == 4:
			image = image[:, :, :-1]

	# normalize the image and convert it to tensor
	normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
	image_processed = preprocess(image)

	return image_processed


def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
	[saved]
	* feat: evaluated deep feature
	'''
	i, image_path, vgg16 = args

	# load image and preprocess the image
	image = skimage.io.imread(image_path)
	image = image.astype('float')
	image = skimage.transform.resize(image, (224, 224))
	image_tensor = preprocess_image(image).unsqueeze(0)

	# get the fc7 layer (second classifier layer)
	layer = list(vgg16.classifier.children())[-2]
	# create the variable
	image_variable = torch.autograd.Variable(image_tensor)

	# to store the fc7 output feature
	feat = torch.zeros((1, 1, 4096))
	def copy_data(model, input, output):
		feat.copy_(output.data)

	vgg16 = vgg16.float()
	# attach copy function to fc7 layer
	h = layer.register_forward_hook(copy_data)
	vgg16(image_variable)
	# detach the copy function from fc7 layer
	h.remove()

	# convert to numpy
	feat = feat.numpy()[0, 0, :]

	# save the feature
	tmp_dir = '../vgg_tmp'
	if not os.path.exists(tmp_dir):
		os.makedirs(tmp_dir)
	np.save(os.path.join(tmp_dir, str(i) + '.npy'), feat)


def distance_to_set(feature, train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	[dist] = scipy.spatial.distance.cdist(np.array([feature]), train_features, 'euclidean')
	dist *= -1
	return dist
