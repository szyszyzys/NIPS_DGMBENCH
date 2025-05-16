from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import entry.gradient_market.skymask.mytorch as my


def create_masknet(param_list, net_type, ctx):
    nworker = len(param_list)
    if net_type == "cnn":
        masknet = CNNMaskNet(param_list, nworker, ctx)
    elif net_type == "resnet20":
        masknet = ResMaskNet(param_list, nworker, ctx)
    elif net_type == "LR":
        masknet = LRMaskNet(param_list, nworker, ctx)
    elif net_type == "lenet":
        masknet = LeNetMaskNet(param_list, nworker, ctx)
    elif net_type == "cifarcnn":
        masknet = CnnCifarMaskNet(param_list, nworker, ctx)
    else:
        masknet = None

    return masknet


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(30, 50, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(1250, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # x尺寸：(batch_size, image_channels, image_width, image_height)

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x, dim=0)

        return x


class CNNMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(CNNMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv1', my.myconv2d(num_workers, device, [x[2] for x in param_list], [x[3] for x in param_list])),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(2, 2))])
        )

        self.fc1 = my.mylinear(num_workers, device, [x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2 = my.mylinear(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])

    def update(self, param_list):
        self.param_list = param_list
        self.conv1.conv1.update([x[0] for x in param_list], [x[1] for x in param_list])
        self.conv2.conv1.update([x[2] for x in param_list], [x[3] for x in param_list])
        self.fc1.update([x[4] for x in param_list], [x[5] for x in param_list])
        self.fc2.update([x[6] for x in param_list], [x[7] for x in param_list])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x, dim=0)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = F.log_softmax(out, dim=0)
        return out


def ResNet20():
    return ResNet(ResidualBlock)


class ResidualMaskBlock(nn.Module):
    def __init__(self, param_list, num_workers, device, stride=1):
        super(ResidualMaskBlock, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.left = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=stride, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU(),
            my.myconv2d(num_workers, device, [x[5] for x in param_list], padding=1),
            my.mybatch_norm(num_workers, device, [x[6] for x in param_list], [x[7] for x in param_list])
        )
        self.shortcut = nn.Sequential()
        if len(param_list[0]) > 10:
            self.shortcut = nn.Sequential(
                my.myconv2d(num_workers, device, [x[10] for x in param_list], stride=stride),
                my.mybatch_norm(num_workers, device, [x[11] for x in param_list], [x[12] for x in param_list])
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(ResMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device
        self.conv1 = nn.Sequential(
            my.myconv2d(num_workers, device, [x[0] for x in param_list], stride=1, padding=1),
            my.mybatch_norm(num_workers, device, [x[1] for x in param_list], [x[2] for x in param_list]),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualMaskBlock, [x[5:35] for x in param_list], num_workers, device, 3,
                                      stride=1)  # 10+10+10
        self.layer2 = self.make_layer(ResidualMaskBlock, [x[35:70] for x in param_list], num_workers, device, 3,
                                      stride=2)  # 15+10+10
        self.layer3 = self.make_layer(ResidualMaskBlock, [x[70:105] for x in param_list], num_workers, device, 3,
                                      stride=2)  # 15+10+10
        self.fc = my.mylinear(num_workers, device, [x[105] for x in param_list], [x[106] for x in param_list])

    def make_layer(self, block, param_list, num_workers, device, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        head = 0
        for stride in strides:
            if stride == 1:
                layers.append(block([x[head:head + 10] for x in param_list], num_workers, device, stride))
                head += 10
            else:
                layers.append(block([x[head:head + 15] for x in param_list], num_workers, device, stride))
                head += 15

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        out = F.log_softmax(out, dim=0)
        return out


class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(561, 6)

    def forward(self, x):
        y_pred = self.linear(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LRMaskNet(nn.Module):
    def __init__(self, param_list, num_workers, device):
        super(LRMaskNet, self).__init__()

        self.num_workers = num_workers
        self.param_list = param_list
        self.device = device

        self.fc = my.mylinear(num_workers, device, [x[0] for x in param_list], [x[1] for x in param_list])

    def forward(self, x):
        y_pred = self.fc(x)
        out = F.log_softmax(y_pred, dim=0)
        return out


class LeNetMaskNet(nn.Module):
    """MaskNet specifically mirroring the LeNet architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's LeNet model IN ORDER:
                               [conv1.w, conv1.b, conv2.w, conv2.b, fc1.w, fc1.b,
                                fc2.w, fc2.b, fc3.w, fc3.b]
            num_workers: Number of workers.
            device: Torch device.
        """
        super(LeNetMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 10
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for LeNet.")

        # --- Layer Definitions (using my.* layers and hardcoded indices) ---
        # Layer params are passed as lists extracted from worker_param_list
        self.conv1 = my.myconv2d(num_workers, device,
                                 [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list]) # Removed kernel_size
                                 # Add stride, padding etc. IF they are non-default and needed by F.conv2d
        self.conv2 = my.myconv2d(num_workers, device,
                                 [w[2] for w in worker_param_list],
                                 [w[3] for w in worker_param_list]) # Removed kernel_size
                                 # Add stride, padding etc. IF non-default
        # Note: Input size for fc1 depends on image size (16*4*4 for 28x28 MNIST)
        self.fc1 = my.mylinear(num_workers, device,
                               [w[4] for w in worker_param_list],
                               [w[5] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device,
                               [w[6] for w in worker_param_list],
                               [w[7] for w in worker_param_list])
        self.fc3 = my.mylinear(num_workers, device,
                               [w[8] for w in worker_param_list],
                               [w[9] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic LeNet forward pass ---
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 10 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.conv2.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.fc1.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.fc2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.fc3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])


class CnnCifarMaskNet(nn.Module):
    """MaskNet specifically mirroring the CNN_CIFAR architecture."""

    def __init__(self, worker_param_list: List[List[torch.Tensor]], num_workers: int, device: torch.device,
                 num_classes=10):
        """
        Args:
            worker_param_list: List (size=num_workers) where each element is a list
                               of parameters for one worker's CNN_CIFAR model IN ORDER:
                               [conv1.w, conv1.b, bn1.w, bn1.b, conv2.w, conv2.b, bn2.w, bn2.b,
                                conv3.w, conv3.b, bn3.w, bn3.b, fc1.w, fc1.b, fc2.w, fc2.b]
            num_workers: Number of workers.
            device: Torch device.
            num_classes: Number of output classes for fc2.
        """
        super(CnnCifarMaskNet, self).__init__()
        self.num_workers = num_workers
        self.device = device

        # --- Parameter Validation ---
        if len(worker_param_list) != num_workers:
            raise ValueError(f"Expected {num_workers} lists in worker_param_list, got {len(worker_param_list)}")
        expected_params = 16
        if any(len(p) != expected_params for p in worker_param_list):
            raise ValueError(
                f"Each inner list in worker_param_list must have exactly {expected_params} tensors for CNN_CIFAR.")

        # --- Layer Definitions ---
        self.conv1 = my.myconv2d(num_workers, device, [w[0] for w in worker_param_list],
                                 [w[1] for w in worker_param_list], padding=1)
        self.bn1 = my.mybatch_norm(num_workers, device, [w[2] for w in worker_param_list],
                                   [w[3] for w in worker_param_list])
        self.conv2 = my.myconv2d(num_workers, device, [w[4] for w in worker_param_list],
                                 [w[5] for w in worker_param_list], padding=1)
        self.bn2 = my.mybatch_norm(num_workers, device, [w[6] for w in worker_param_list],
                                   [w[7] for w in worker_param_list])
        self.conv3 = my.myconv2d(num_workers, device, [w[8] for w in worker_param_list],
                                 [w[9] for w in worker_param_list], padding=1)
        self.bn3 = my.mybatch_norm(num_workers, device, [w[10] for w in worker_param_list],
                                   [w[11] for w in worker_param_list])
        self.pool = nn.MaxPool2d(2, 2)  # Standard layer
        # Note: Input features for fc1 depends on pooling (256 * 4 * 4 for CIFAR 32->16->8->4)
        self.fc1 = my.mylinear(num_workers, device, [w[12] for w in worker_param_list],
                               [w[13] for w in worker_param_list])
        self.fc2 = my.mylinear(num_workers, device, [w[14] for w in worker_param_list],
                               [w[15] for w in worker_param_list])

    def forward(self, x):
        # --- Mimic CNN_CIFAR forward pass ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Optional: Apply LogSoftmax if MaskNet is used for classification during training
        # x = F.log_softmax(x, dim=1)
        return x

    def update(self, worker_param_list: List[List[torch.Tensor]]):
        """Updates the internal lists of worker parameters."""
        if len(worker_param_list) != self.num_workers or any(len(p) != 16 for p in worker_param_list):
            raise ValueError("Invalid parameter list structure for update.")
        # Call update on each my.* layer
        self.conv1.update([w[0] for w in worker_param_list], [w[1] for w in worker_param_list])
        self.bn1.update([w[2] for w in worker_param_list], [w[3] for w in worker_param_list])
        self.conv2.update([w[4] for w in worker_param_list], [w[5] for w in worker_param_list])
        self.bn2.update([w[6] for w in worker_param_list], [w[7] for w in worker_param_list])
        self.conv3.update([w[8] for w in worker_param_list], [w[9] for w in worker_param_list])
        self.bn3.update([w[10] for w in worker_param_list], [w[11] for w in worker_param_list])
        self.fc1.update([w[12] for w in worker_param_list], [w[13] for w in worker_param_list])
        self.fc2.update([w[14] for w in worker_param_list], [w[15] for w in worker_param_list])
