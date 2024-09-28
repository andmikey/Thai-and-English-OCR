from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

# Hardcode based on contents of reference file
# This means the model will output a 286-length tensor for predictions
# Where there is not a coresponding class, the entry for that class will just be zero
NUM_CLASSES = 286


class ImageDataset(Dataset):
    def __init__(self, data_files: List[Path]):
        self.data = pd.concat(
            [
                pd.read_csv(
                    data_file,
                    header=None,
                    names=["language", "dpi", "style", "label", "path"],
                )
                for data_file in data_files
            ]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        image_path = self.data.iloc[idx]["path"]
        # .float() converts from binary array to float array
        image = pil_to_tensor(Image.open(image_path)).float()
        # Resize all images to 64x64
        resized_image = Resize(size=(64, 64))(image)
        label = self.data.iloc[idx]["label"]
        return resized_image, label


class Dataset:
    def __init__(self, paths, batches):
        self.dataset = ImageDataset(paths)
        self.loader = DataLoader(self.dataset, batch_size=batches, shuffle=True)

    def info(self):
        self.points = len(self.dataset)


def load_datasets(dataset_paths: List[Path], batches):
    return Dataset(dataset_paths, batches)


class Evaluator:
    def __init__(self, model, input, device):
        self.model = model
        self.input = input
        self.device = device

    def run_on_input(self):
        # Keep a list-of-lists then flatten it later
        overall_true_labels = []
        overall_predicted_labels = []

        for idx, data in enumerate(self.input.loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            predicted = torch.argmax(self.model(inputs), dim=1)
            overall_true_labels.append(labels)
            overall_predicted_labels.append(predicted)

        self.actual = torch.cat(overall_true_labels).cpu()
        self.predicted = torch.cat(overall_predicted_labels).cpu()
        self.calculate_metrics()

    def calculate_metrics(self):
        self.precision = precision_score(self.predicted, self.actual, average="macro")
        self.recall = recall_score(self.predicted, self.actual, average="macro")
        self.f1 = f1_score(self.predicted, self.actual, average="macro")
        self.accuracy = accuracy_score(self.predicted, self.actual)

    def __str__(self):
        output_str = f"""
Metric | Value 
---|---
Precision | {self.precision:.3%}
Recall | {self.recall:.3%}
F1 | {self.f1:.3%}
Accuracy | {self.accuracy:.3%}"""

        return output_str
