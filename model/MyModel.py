import torch.nn as nn
from torchvision import models
import torch

# class ResNet18(nn.Module):
#     def __init__(self, num_class):
#         super(ResNet18, self).__init__()
#         self.resnet18 = models.resnet18(pretrained=True)
#         in_features = self.resnet18.fc.in_features
#         self.resnet18.fc = nn.Linear(in_features, num_class)
#
#     def forward(self, x1, x2):
#         out1 = self.resnet18(x1)
#         out2 = self.resnet18(x2)
#
#         return out1, out2

class ResNet18(nn.Module):
    def __init__(self, num_class, dropout_prob):
        super(ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)

        # 截断全连接层之前的部分
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

        # 添加自定义的全连接层，其中包含 Dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1000, num_class)
        )

    def forward(self, x1, x2):
        ft1 = self.features(x1)
        ft2 = self.features(x2)

        out1 = self.classifier(ft1)
        out2 = self.classifier(ft2)

        return out1, out2

class ResNet50(nn.Module):
    def __init__(self, num_class, dropout_prob):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)

        # 截断全连接层之前的部分
        self.features = nn.Sequential(*list(resnet50.children())[:-2])

        # 添加自定义的全连接层，其中包含 Dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_class)
        )

    def forward(self, x1, x2):
        ft1 = self.features(x1)
        ft2 = self.features(x2)

        out1 = self.classifier(ft1)
        out2 = self.classifier(ft2)

        return out1, out2

class Inceptionv3(nn.Module):
    def __init__(self, num_class, dropout_prob):
        super(Inceptionv3, self).__init__()

        self.inception = models.inception_v3(pretrained=True)

        # 修改分类层以适应新的类别数量
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_class)

    def forward(self, x1, x2):
        out1 = self.inception(x1)
        out2 = self.inception(x2)

        return out1, out2

class VGG19(nn.Module):
    def __init__(self, num_classes, dropout_prob):
        super(VGG19, self).__init__()

        self.vgg19 = nn.Sequential(
            self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                               512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),  # BatchNorm added here
            nn.Dropout(dropout_prob),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),  # BatchNorm added here
            nn.Dropout(dropout_prob),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x1, x2):
        out1 = self.vgg19(x1)
        out2 = self.vgg19(x2)

        return out1, out2

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3  # 输入图像的通道数

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)