from pathlib import Path

import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, data_file: Path):
        self.data = pd.read_csv(
            data_file, header=None, names=["language", "dpi", "style", "label", "path"]
        )

    def __len__(self):
        self.data.shape[0]

    def __getitem__(self, idx: int):
        image_path = self.data[idx]["class"]
        image = read_image(image_path)
        label = self.data[idx]["label"]
        return image, label


class BasicNetwork(nn.module):
    def __init__(self):
        super(BasicNetwork, self).__init__(self)
        # TODO some definitions here

    def forward(self, input):
        # TODO return something here
        # Example net here: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        # And here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        output = _
        return output
