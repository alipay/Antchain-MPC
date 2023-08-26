# -*- coding: utf-8 -*-
"""train_resnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ABY48-UAjltSOtMYgbxUPWI6sS_BQ-Yq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

"""## Lets get the data, model and setup trainnig code"""

train_loader = DataLoader(datasets.CIFAR10("./", train=True, transform=transforms.ToTensor(), download=True), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.CIFAR10("./", train=False, transform=transforms.ToTensor(), download=True), batch_size=128, shuffle=True)

print(f"Training images {len(train_loader.dataset)}, Test images {len(test_loader.dataset)}")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer = conv_layer

        self.conv1 = conv_layer(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = linear_layer(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# NOTE: Only supporting default (kaiming_init) initializaition.
def resnet18(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for Resnets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)

model = resnet18(nn.Conv2d, nn.Linear, "kaiming_normal").cuda()
# print(model)

epochs = 5
lr = 0.1
lr_set = [3, 4, 6, 8, 10]
lr_set = [2 ** -x for x in lr_set]
#lr_set = [0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
#lr_set = [0.0078125, 0.015625, 0.25, 0.0625, 0.015625]

optimizer = optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
lrs = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

"""### let's start training"""

def get_acc(model, loader):
  correct = 0
  total = 0
  for img, label in loader:
    correct += torch.sum(torch.argmax(model(img.cuda()), -1).cpu() == label).item()
    total += len(img)
  return 100*correct/total

for e in range(epochs):
  optimizer.param_groups[0]["lr"] = lr_set[e]
  print("lr", optimizer.param_groups[0]["lr"])
  for img, label in train_loader:
    out = model(img.cuda())
    
    optimizer.zero_grad()
    loss = criterion(out, label.cuda())
    loss.backward()
    optimizer.step()
  lrs.step()
  print(f"Epoch {e}, training accuracy {get_acc(model, train_loader)}, test accuracy {get_acc(model, test_loader)}")