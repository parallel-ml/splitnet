import torch
import torch.nn as nn


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

    def __init__(self, num_classes=1000, splits=(2, 4, 8)):
        super(SplitAlexNet, self).__init__()

        if len(splits) == 3:
            self.splits = [1, 1] + list(splits)
        else:
            self.splits = splits

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.ModuleList(
            [nn.Conv2d(384 // self.splits[0], 256 // self.splits[0], kernel_size=3, padding=1) for _ in
             range(self.splits[0])])
        self.conv2 = nn.ModuleList(
            [nn.Conv2d(256 // self.splits[1], 256 // self.splits[1], kernel_size=3, padding=1) for _ in
             range(self.splits[1])])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.ModuleList(
            [nn.Linear(256 * 6 * 6 // self.splits[2], 4096 // self.splits[2]) for _ in range(self.splits[2])])
        self.fc2 = nn.ModuleList(
            [nn.Linear(4096 // self.splits[3], 4096 // self.splits[3]) for _ in range(self.splits[3])])
        self.fc3 = nn.ModuleList(
            [nn.Linear(4096 // self.splits[4], num_classes // self.splits[4]) for _ in range(self.splits[4])])

    def forward(self, x):
        x = self.features(x)
        x = [x]

        x = self._scale(x, 0)
        x = [mod(x[i]) for i, mod in enumerate(self.conv1)]

        x = self._scale(x, 1)
        x = [mod(x[i]) for i, mod in enumerate(self.conv2)]

        x = torch.cat(x, dim=1)
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = [x]

        x = self._scale(x, 2)
        x = [mod(x[i]) for i, mod in enumerate(self.fc1)]

        x = self._scale(x, 3)
        x = [mod(x[i]) for i, mod in enumerate(self.fc2)]

        x = self._scale(x, 4)
        x = [mod(x[i]) for i, mod in enumerate(self.fc3)]
        return x

    def _scale(self, x, scale):
        split = self.splits[scale]
        tensor = torch.cat(x, dim=1)
        chunk_size = tensor.size(1) // split
        return list(torch.split(tensor, chunk_size, 1))


if __name__ == '__main__':
    from torchsummary import summary

    net = SplitAlexNet(1000, (2, 2, 2, 2, 2))
    summary(net, (3, 224, 224))
