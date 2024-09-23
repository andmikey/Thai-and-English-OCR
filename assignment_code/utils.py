from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# from torcheval.metrics import functional as torch_eval
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor


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
        # .long() converts from binary array to float array
        image = pil_to_tensor(Image.open(image_path)).float()
        # Resize all images to 64x64
        resized_image = Resize(size=(64, 64))(image)
        label = self.data.iloc[idx]["label"]
        return resized_image, label


class Dataset:
    def __init__(self, paths, batches):
        self.dataset = ImageDataset(paths)
        self.loader = DataLoader(self.dataset, batch_size=batches)

    def info(self):
        self.points = len(self.dataset)


def load_datasets(dataset_paths: List[Path], batches):
    return Dataset(dataset_paths, batches)


# class Evaluator:
#     def __init__(self, predicted, actual):
#         self.predicted = predicted
#         self.actual = actual

#     def calculate_metrics(self):
#         self.precision = torch_eval.multiclass_precision(self.predicted, self.actual)
#         self.recall = torch_eval.multiclass_recall(self.predicted, self.actual)
#         self.f1 = torch_eval.multiclass_f1_score(self.predicted, self.actual)
#         self.accuracy = torch_eval.multiclass_accuracy(self.predicted, self.actual)
#         self.auc = torch_eval.multiclass_auroc(self.predicted, self.actual)
#         self.roc = torch_eval.multiclass_precision_recall_curve(
#             self.predicted, self.actual
#         )

#     def __str__(self):
#         output_str = f"""Precision: {self.precision}
# Recall: {self.recall}
# F1: {self.f1}
# Accuracy: {self.accuracy}
# AUC: {self.auc}"""

#         return output_str
