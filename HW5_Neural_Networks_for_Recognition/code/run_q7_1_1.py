import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device('cpu')

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.01
hidden_size = 64

train_x_ts = torch.from_numpy(train_x).type(torch.float32)
train_y_ts = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=batch_size, shuffle=False)


class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


model = TwoLayerNet(train_x.shape[1], hidden_size, train_y.shape[1])

train_loss = []
train_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
for itr in range(max_iters):
    total_loss = 0
    correct = 0
    for data in train_loader:
        # get the inputs
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]

        # get output
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, targets)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += predicted.eq(targets.data).cpu().sum().item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/train_y.shape[0]
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

torch.save(model.state_dict(), "q7_1_1_model_parameter.pkl")


# run on test data
test_correct = 0
for data in test_loader:
    # get the inputs
    inputs = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])
    targets = torch.max(labels, 1)[1]

    # get output
    y_pred = model(inputs)
    loss = nn.functional.cross_entropy(y_pred, targets)

    predicted = torch.max(y_pred, 1)[1]
    test_correct += predicted.eq(targets.data).cpu().sum().item()

test_acc = test_correct/test_y.shape[0]

print('Test accuracy: {}'.format(test_acc))
