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

device = torch.device('cpu')

max_iters = 50
# pick a batch size, learning rate
batch_size = 64
learning_rate = 0.01

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(14*14*4, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*4)
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

torch.save(model.state_dict(), "q7_1_2_model_parameter.pkl")

# checkpoint = torch.load('q7_1_2_model_parameter.pkl')
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
