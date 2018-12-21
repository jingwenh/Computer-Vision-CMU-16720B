import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
data_mean = np.sum(train_x, axis=0)/train_x.shape[0]
train_x -= data_mean

u, s, vh = np.linalg.svd(train_x)
P = vh[:dim, :]

# rebuild a low-rank version
lrank = np.dot(train_x, np.transpose(P))

# rebuild it
recon = np.dot(lrank, P)
recon += data_mean
train_x += data_mean

train_psnr = 0
for i in range(recon.shape[0]):
    train_psnr += psnr(train_x[i], recon[i])
train_psnr /= recon.shape[0]
print(train_psnr)

for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()

# build valid dataset
valid_mean = np.sum(valid_x, axis=0)/valid_x.shape[0]
valid_x -= valid_mean
recon_valid = np.dot(np.dot(valid_x, np.transpose(P)), P)
recon_valid += valid_mean
valid_x += valid_mean

total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())

selected_inds = [4, 50, 500, 520, 1740, 1720, 2000, 2010, 3400, 3420]

for i in selected_inds:
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon_valid[i].reshape(32,32).T)
    plt.show()