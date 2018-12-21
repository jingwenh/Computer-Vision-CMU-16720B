import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt

device = torch.device('cpu')

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 0.01

train_data = scipy.io.loadmat('../data/nist36_train.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x = np.array([np.reshape(x, (32, 32)) for x in train_x])
test_x = np.array([np.reshape(x, (32, 32)) for x in test_x])

train_x_ts = torch.from_numpy(train_x).type(torch.float32).unsqueeze(1)
train_y_ts = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(16*16*16, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 36))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 16*16*16)
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
        targets = torch.max(labels, 1)[1]

        # get output
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, targets)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == targets.data).item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/train_x.shape[0]
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

torch.save(model.state_dict(), "q7_1_3_model_parameter.pkl")

# checkpoint = torch.load('q7_1_3_model_parameter.pkl')
# model.load_state_dict(checkpoint)

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
    test_correct += torch.sum(predicted == targets.data).item()

test_acc = test_correct/test_x.shape[0]

print('Test accuracy: {}'.format(test_acc))
