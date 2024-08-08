import paddle
import paddle.optimizer
from paddle.vision.transforms import Compose, Normalize, Resize
import paddle.nn.functional as F
from paddle.vision.image import image_load
from paddle.vision.datasets import MNIST
import paddle.nn as nn
import paddle.metric as metric
import os
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集与数据归一化
tranform = Compose([
    Resize(size=[32, 32], interpolation='bilinear'),
    Normalize(mean=[127.5], std=[127.5], data_format='CHW')
])

train_dataset = MNIST(mode='train', transform=tranform, backend='cv2')
test_dataset = MNIST(mode='test', transform=tranform, backend='cv2')


# 定义模型
class LeNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=6,
                               kernel_size=5, stride=1, padding=2)
        self.maxPool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=6, out_channels=16,
                               kernel_size=5, stride=1, padding=0)
        self.maxPool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=120,
                               kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(in_features=480, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


# 训练模型
model = paddle.Model(LeNet())
optim = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=model.parameters())
model.prepare(optim, loss=nn.CrossEntropyLoss(),
              metrics=metric.Accuracy())
model.fit(train_dataset, batch_size=64, epochs=10, verbose=2)
