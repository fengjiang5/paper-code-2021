import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import sampler
import time

# data = get_CIFAR10_data()
# for k, v in data.items():
#     print('%s:'%k, v.shape)


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel=10, kernel_size=3,
                 padding=1, stride=1):
        super(Conv, self).__init__()
        self.c1 = nn.Conv2d(in_channel, out_channels=16, kernel_size=kernel_size,
                            stride=stride, padding=padding)  # !!!  # 6*32*32
        self.r2 = nn.ReLU(inplace=True)
        self.s3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6*16*16
        self.c4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size,
                            stride=stride, padding=padding)  # !!!  # 6*16*16
        self.r5 = nn.ReLU(inplace=True)
        self.s6 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6*8*8
        self.f7 = nn.Linear(32*8*8, 96)
        self.r8 = nn.ReLU()
        self.f9 = nn.Linear(96, 10)

    def forward(self, x):
        out = self.s3(self.r2(self.c1(x)))
        out = self.s6(self.r5(self.c4(out)))
        out = self.f7(out.view(-1, 32*8*8))
        out = self.f9(self.r8(out))
        return out


class FcNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.r1(out)
        out = self.fc2(out)
        out = self.fc3(self.r2(out))
        return out


NUM_TRAIN = 49000

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv(in_channel=3).to(device)
model = model.train()
criterion = nn.CrossEntropyLoss()
optim1 = optim.Adam(model.parameters(), lr=1e-5)

total_step = len(loader_train)

losses = []
accs = []

time_start1 = time.time()
for epoch in range(10):
    for i, (image, label) in enumerate(loader_train):
        image = image.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        scores = model(image)
        loss = criterion(scores, label)

        optim1.zero_grad()
        loss.backward()
        optim1.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/10], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, i + 1, total_step, loss.item()))
            losses.append(loss.item())

    with torch.no_grad():
        correct = 0
        total = 0
        model = model.eval()
        for images, labels in loader_test:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

        accuracy = correct / total
        accs.append(accuracy)
        print('Accuracy of the network on the test images: {} %'.format(100 * accuracy))

    model = model.train()
time_end1 = time.time()

model2 = FcNet(input_size=3*32*32, hidden_size=96, output_size=10).to(device)
model2 = model2.train()
criterion2 = nn.CrossEntropyLoss()
optim2 = optim.Adam(model2.parameters(), lr=1e-5)

total_step2 = len(loader_train)

losses2 = []
accs2 = []
time_start2 = time.time()
for epoch in range(10):
    for i, (image, label) in enumerate(loader_train):
        image = image.reshape(-1, 3*32*32).to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)

        scores2 = model2(image)
        loss2 = criterion2(scores2, label)

        optim2.zero_grad()
        loss2.backward()
        optim2.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/10], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, i + 1, total_step2, loss2.item()))
            losses2.append(loss2.item())

    with torch.no_grad():
        correct2 = 0
        total2 = 0
        model2 = model2.eval()
        for images, labels in loader_test:
            images = images.reshape(-1, 3*32*32).to(device)
            labels = labels.to(device)
            outputs2 = model2(images)
            _, predict2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (predict2 == labels).sum().item()

        accuracy2 = correct2 / total2
        accs2.append(accuracy2)
        print('Accuracy of the network on the test images: {} %'.format(100 * accuracy2))

    model2 = model2.train()

time_end2 = time.time()


plt.figure()
plt.title('loss')
plt.plot(losses)
plt.plot(losses2)
plt.legend(['Conv', 'Fully-Net'])

plt.figure()
plt.title('accuracy')
plt.plot(accs)
plt.plot(accs2)
plt.legend(['Conv', 'Fully-Net'])
plt.show()
print(time_end1-time_start1, time_end2-time_start2)

# def test_size():
#     x = torch.zeros((64, 3, 32, 32), dtype=torch.float32)
#     model = Conv(in_channel=3, out_channel=10)
#     scores = model(x)
#     print(scores.size())


# test_size()
