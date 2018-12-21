import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import multiprocessing

def get_num_CPU():
	'''
	Counts the number of CPUs available in the machine.
	'''
	return multiprocessing.cpu_count()

def get_VGG16_weights():
	'''
	Returns the weights and meta-information of the VGG-16 network.

	[output]
	* vgg16_weights: numpy.ndarray of shape (L,3)
					 The first column stores the type of network layer.
					 If the type is "conv2d" or "linear", the second and third column stores the weight and bias.
					 If the type is "maxpool2d", the second column stores the kernel size (assuming the same stride size).
	'''

	vgg16 = torchvision.models.vgg16(pretrained=True)
	vgg16_weights = []

	for layer in vgg16.features:
		if isinstance(layer,torch.nn.Conv2d):
			l = ["conv2d",layer.weight.detach().numpy(),layer.bias.detach().numpy()]
		if isinstance(layer,torch.nn.ReLU):
			l = ["relu"]
		if isinstance(layer,torch.nn.MaxPool2d):
			l = ["maxpool2d",layer.kernel_size]
		vgg16_weights.append(l)
	for layer in vgg16.classifier:
		if isinstance(layer,torch.nn.Linear):
			l = ["linear",layer.weight.detach().numpy(),layer.bias.detach().numpy()]
		if isinstance(layer,torch.nn.ReLU):
			l = ["relu"]
		if isinstance(layer,torch.nn.Dropout): continue
		vgg16_weights.append(l)

	return vgg16_weights

def display_filter_responses(response_maps):
	'''
	Visualizes the filter response maps.

	[input]
	* response_maps: a numpy.ndarray of shape (H,W,3F)
	'''
	
	fig = plt.figure(1)
	
	for i in range(20):
		plt.subplot(5,4,i+1)
		resp = response_maps[:,:,i*3:i*3+3]
		resp_min = resp.min(axis=(0,1),keepdims=True)
		resp_max = resp.max(axis=(0,1),keepdims=True)
		resp = (resp-resp_min)/(resp_max-resp_min)
		plt.imshow(resp)
		plt.axis("off")

	plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
	plt.show()

def save_wordmap(wordmap, filename):
	fig = plt.figure(2)
	plt.axis('equal')
	plt.axis('off')
	plt.imshow(wordmap)
	plt.savefig(filename, pad_inches=0)
