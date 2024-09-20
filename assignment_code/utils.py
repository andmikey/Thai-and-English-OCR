from model import ImageDataset
from torch.utils import DataLoader
from torcheval.metrics import functional as torch_eval


class Dataset:
    def __init__(self, path, batches):
        self.dataset = ImageDataset(path)
        self.loader = DataLoader(self.dataset, batch_size=batches)


def load_datasets(dataset_path, batches):
    training_dataset = Dataset(dataset_path / "training_set.txt", batches)
    testing_dataset = Dataset(dataset_path / "testing_set.txt", batches)
    validation_dataset = Dataset(dataset_path / "validation_set.txt", batches)

    return training_dataset, testing_dataset, validation_dataset


class Evaluator:
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual

    def calculate_metrics(self):
        self.precision = torch_eval.multiclass_precision(self.predicted, self.actual)
        self.recall = torch_eval.multiclass_recall(self.predicted, self.actual)
        self.f1 = torch_eval.multiclass_f1_score(self.predicted, self.actual)
        self.accuracy = torch_eval.multiclass_accuracy(self.predicted, self.actual)
        self.auc = torch_eval.multiclass_auroc(self.predicted, self.actual)
        self.roc = torch_eval.multiclass_precision_recall_curve(
            self.predicted, self.actual
        )

    def __str__(self):
        output_str = f"""Precision: {self.precision}
Recall: {self.recall}
F1: {self.f1}
Accuracy: {self.accuracy}
AUC: {self.auc}"""

        return output_str
