import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


train_loader = DataLoader(
    datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,)),
        ]),
    ),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=64,
    shuffle=True
)

def imgShow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        # 在 (2, 2) 的窗口上做池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将维度转成以batch为第一维 剩余维数相乘为第二维
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1: ] # 第一个维度的batch不考虑
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



#创建优化器
import torch.optim as optim


if __name__ == '__main__':
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # # 展示图片
    # imgShow(torchvision.utils.make_grid(images))

    net = Net()

    optimizer = optim.SGD(net.parameters(), lr=0.01)  # lr代表学习率
    criterion = nn.CrossEntropyLoss()

    def train(epoch):
        net.train()
        running_loss = 0.0
        for j in range(epoch):
            for i, data in enumerate(train_loader):
                # 得到输入和标签
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 打印日志
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[{0}, {1}] loss: {2}'.format(j + 1, i + 1, running_loss / 100))
                    running_loss = 0.0


    train(20)

    # 观察模型预测效果
    correct = 0
    total = 0
    with torch.no_grad():  # 或者model.eval()
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
