import torch
import torch.nn as nn

from util import param_size


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SplitAlexNet(nn.Module):

    def __init__(self, num_classes=1000, split=(2, 4, 8)):
        super(SplitAlexNet, self).__init__()

        assert len(split) == 3, 'Split must be length of 3.'

        self.scale0 = split[0]
        self.scale1 = split[1] // split[0]
        self.scale2 = split[2] // split[1]

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.ModuleList([nn.Linear(256 * 6 * 6 // split[0], 4096 // split[0]) for _ in range(split[0])])
        self.fc2 = nn.ModuleList([nn.Linear(4096 // split[1], 4096 // split[1]) for _ in range(split[1])])
        self.fc3 = nn.ModuleList([nn.Linear(4096 // split[2], num_classes // split[2]) for _ in range(split[2])])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = [x]

        x = self._scale(x, self.scale0)
        x = [mod(x[i]) for i, mod in enumerate(self.fc1)]

        x = self._scale(x, self.scale1)
        x = [mod(x[i]) for i, mod in enumerate(self.fc2)]

        x = self._scale(x, self.scale2)
        x = [mod(x[i]) for i, mod in enumerate(self.fc3)]
        return x

    def _scale(self, x, scale):
        chunk_size = x[0].size(-1) // scale

        scale_size = []
        for i in range(scale):
            scale_size.append(i * chunk_size)
        scale_size.append(-1)

        x_scale = []
        for x_unit in x:
            x_unit = torch.split(x_unit, chunk_size, dim=-1)
            x_scale += list(x_unit)
        return x_scale


if __name__ == '__main__':
    from torchsummary import summary

    net = SplitAlexNet(1000, (1, 1, 3))
    summary(net, (3, 224, 224))
