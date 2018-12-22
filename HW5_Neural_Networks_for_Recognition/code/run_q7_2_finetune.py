import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

device = torch.device('cpu')

train_dir = '../data/oxford-flowers17/train'
val_dir = '../data/oxford-flowers17/val'
test_dir = '../data/oxford-flowers17/test'

batch_size = 32
num_workers = 4
num_epochs1 = 10
num_epochs2 = 10
learning_rate1 = 0.001
learning_rate2 = 1e-5

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def run_epoch(model, loader, optimizer):
    # Set the model to training mode
    model.train()
    total_loss = 0
    for x, y in loader:
        inputs = torch.autograd.Variable(x.type(torch.FloatTensor))
        labels = torch.autograd.Variable(y.type(torch.FloatTensor).long())

        # forward
        y_pred = model(inputs)
        loss = nn.functional.cross_entropy(y_pred, labels)
        total_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss


def calc_accuracy(model, loader):
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0
    for x, y in loader:
        input = torch.autograd.Variable(x.type(torch.FloatTensor), volatile=True)
        labels = torch.autograd.Variable(y)

        # forward
        y_pred = model(input)
        predicted = torch.max(y_pred, 1)[1]
        num_correct += torch.sum(predicted == labels.data).item()
        num_samples += x.size(0)

    # Calculate the accuracy
    acc = num_correct/num_samples
    return acc


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

model = torchvision.models.squeezenet1_1(pretrained=True)
num_classes = len(trainset.classes)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
model.num_classes = num_classes

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate1)
print('===== train stage 1 =====')
train_accs = []
train_losses = []
for epoch in range(num_epochs1):
    # Run an epoch over the training data.
    train_loss = run_epoch(model, train_loader, optimizer)

    # Check accuracy on the train and val sets.
    train_acc = calc_accuracy(model, train_loader)
    val_acc = calc_accuracy(model, val_loader)

    train_accs.append(train_acc)
    train_losses.append(train_loss)

    print("itr: {:02d} \t train_loss: {:.4f} \t train_acc: {:.4f} \t val_acc: {:.4f}".format(epoch, train_loss, train_acc, val_acc))

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate2)
print('===== train stage 2 =====')
for epoch in range(num_epochs2):
    train_loss = run_epoch(model, train_loader, optimizer)

    train_acc = calc_accuracy(model, train_loader)
    val_acc = calc_accuracy(model, val_loader)

    train_accs.append(train_acc)
    train_losses.append(train_loss)

    print("itr: {:02d} \t train_loss: {:.4f} \t train_acc: {:.4f} \t val_acc: {:.4f}".format(epoch, train_loss, train_acc, val_acc))

plt.figure('accuracy')
plt.plot(range(20), train_accs, color='g')
plt.legend(['train accuracy'])
plt.show()

plt.figure('loss')
plt.plot(range(20), train_losses, color='g')
plt.legend(['train loss'])
plt.show()

torch.save(model.state_dict(), "q7_2_finetune_model_parameter.pkl")

# checkpoint = torch.load('q7_2_finetune_model_parameter.pkl')
# model.load_state_dict(checkpoint)

test_acc = calc_accuracy(model, test_loader)
print("Test Accuracy: {:.4f}".format(test_acc))
