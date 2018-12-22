import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

train_dir = '../data/oxford-flowers17/train'
val_dir = '../data/oxford-flowers17/val'
test_dir = '../data/oxford-flowers17/test'

batch_size = 32
num_workers = 4
max_iters = 200
learning_rate = 0.001

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
trainset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(trainset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True)

val_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
valset = ImageFolder(val_dir, transform=val_transform)
val_loader = DataLoader(valset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)

test_transform = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
testset = ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(testset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=7, stride=1, padding=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(stride=2, kernel_size=2)
                                   )
        self.fc1 = nn.Sequential(nn.Linear(28*28*120, 84),
                                 nn.ReLU()
                                 )
        self.fc2 = nn.Sequential(nn.Linear(84, 17))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 28*28*120)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def calc_accuracy(loader):
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0
    for x, y in loader:
        inputs = torch.autograd.Variable(x.type(torch.FloatTensor), volatile=True)
        labels = torch.autograd.Variable(y)

        # forward
        y_pred = model(inputs)
        predicted = torch.max(y_pred, 1)[1]
        num_correct += torch.sum(predicted == labels.data).item()
        num_samples += x.size(0)

    # Calculate the accuracy
    acc = num_correct/num_samples
    return acc


model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loss = []
train_acc = []
val_acc = []
for itr in range(81, max_iters):
    total_loss = 0
    correct = 0
    model.train()
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

    valid_acc = calc_accuracy(val_loader)
    val_acc.append(valid_acc)

    print("itr: {:02d} \t train_loss: {:.4f} \t train_acc: {:.4f} \t val_acc: {:.4f}".format(itr, total_loss, acc, valid_acc))

    if itr%10 == 0:
        torch.save(model.state_dict(), "q7_2_scratch_model_parameter"+str(itr)+".pkl")
    if itr == 100:
        learning_rate /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# checkpoint = torch.load('q7_2_scratch_model_parameter200.pkl')
# model.load_state_dict(checkpoint)

test_acc = calc_accuracy(test_loader)
print("Test Accuracy: {:.4f}".format(test_acc))
