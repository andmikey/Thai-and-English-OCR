import torch.nn as nn


class BasicNetwork(nn.Module):
    def __init__(self, num_classes, img_dims):
        # Some borrowing from this, adjusted to the right sizes for a 64x64 image:
        # https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37
        super().__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        avgp = nn.AvgPool2d(kernel_size=4, stride=2)
        conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        fl = nn.Flatten()
        fc = nn.Linear(in_features=12 * 12 * 16, out_features=120)
        tanh = nn.Tanh()
        fc2 = nn.Linear(in_features=120, out_features=84)
        fc3 = nn.Linear(in_features=84, out_features=num_classes)
        # Adjust output so all probabilities sum to 1
        softmax = nn.Softmax()

        self.feature = nn.Sequential(conv1, tanh, avgp, conv2, tanh, avgp)
        self.classifier = nn.Sequential(fl, fc, tanh, fc2, tanh, fc3, softmax)

    def forward(self, x):
        # Needs to return tensor in the shape (num_classes)
        # Criterion CrossEntropyLoss takes class index
        return self.classifier(self.feature(x))
