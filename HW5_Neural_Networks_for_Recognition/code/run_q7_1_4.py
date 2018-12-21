import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches

from q4 import *

device = torch.device('cpu')

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 0.01

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True,
                                      download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False,
                                     download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(7*7*32, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 47))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        return x


model = Net()

train_loss = []
train_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for itr in range(max_iters):
    total_loss = 0
    correct = 0
    for data in train_loader:
        # get the inputs
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])

        # get output
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, labels)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == labels.data).item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/len(trainset)
    train_loss.append(total_loss)
    train_acc.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

plt.figure('accuracy')
plt.plot(range(max_iters), train_acc, color='g')
plt.legend(['train accuracy'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='g')
plt.legend(['train loss'])
plt.show()

print('Train accuracy: {}'.format(train_acc[-1]))

torch.save(model.state_dict(), "q7_1_4_model_parameter.pkl")

# checkpoint = torch.load('q7_1_4_model_parameter.pkl')
# model.load_state_dict(checkpoint)

# run on test data
test_correct = 0
for data in test_loader:
    # get the inputs
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])

    # get output
    y_pred = model(inputs)
    loss = nn.functional.cross_entropy(y_pred, labels)

    predicted = torch.max(y_pred, 1)[1]
    test_correct += torch.sum(predicted == labels.data).item()

test_acc = test_correct/len(testset)

print('Test accuracy: {}'.format(test_acc))


# ======== Evaluate it on the findLetters bounded boxes. ======== #
def get_data_from_image(img_path):
    im1 = skimage.img_as_float(skimage.io.imread(img_path))
    bboxes, bw = findLetters(im1)

    # find the rows using..RANSAC, counting, clustering, etc.
    heights = [bbox[2] - bbox[0] for bbox in bboxes]
    mean_height = sum(heights) / len(heights)
    # sort the bounding boxes with center y
    centers = [((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2, bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox
               in
               bboxes]
    centers = sorted(centers, key=lambda p: p[0])
    rows = []
    pre_h = centers[0][0]
    # cluster rows
    row = []
    for c in centers:
        if c[0] > pre_h + mean_height:
            row = sorted(row, key=lambda p: p[1])
            rows.append(row)
            row = [c]
            pre_h = c[0]
        else:
            row.append(c)
    row = sorted(row, key=lambda p: p[1])
    rows.append(row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    data = []
    for row in rows:
        row_data = []
        for y, x, h, w in row:
            # crop out the character
            crop = bw[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
            # pad it to square
            h_pad, w_pad = 0, 0
            if h > w:
                h_pad = h // 20
                w_pad = (h - w) // 2 + h_pad
            elif h < w:
                w_pad = w // 20
                h_pad = (w - h) // 2 + w_pad
            crop = np.pad(crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            # resize to 28*28
            crop = skimage.transform.resize(crop, (28, 28))
            crop = skimage.morphology.erosion(crop, kernel)
            crop = np.transpose(crop)
            crop = 1-crop
            # plt.figure()
            # plt.imshow(crop)
            # plt.show()

            row_data.append(crop)
        data.append(np.array(row_data))

    return data


ind2c = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
         40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}


def detect(input_data):
    tss = []
    for item in input_data:
        ts = transform(np.expand_dims(item, axis=2)).type(torch.float32)
        tss.append(ts)
    x_ts = torch.stack(tss, dim=0)

    # get the inputs
    inputs = torch.autograd.Variable(x_ts)

    # get output
    y_pred = model(inputs)

    predicted = torch.max(y_pred, 1)[1]

    result = predicted.numpy()
    row_s = ''
    for i in range(result.shape[0]):
        row_s += ind2c[int(result[i])]

    print(row_s)


def detect_for_images():
    for img in os.listdir('../images'):
        img_path = os.path.join('../images',img)
        input_data = get_data_from_image(img_path)
        for row_data in input_data:
            detect(row_data)


detect_for_images()
