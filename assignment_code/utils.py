from model import ImageDataset
from torch.utils import DataLoader


class Dataset:
    def __init__(self, path, batches):
        self.dataset = ImageDataset(path)
        self.loader = DataLoader(self.dataset, batch_size=batches)


def load_datasets(dataset_path, batches):
    training_dataset = Dataset(dataset_path / "training_set.txt", batches)
    testing_dataset = Dataset(dataset_path / "testing_set.txt", batches)
    validation_dataset = Dataset(dataset_path / "validation_set.txt", batches)

    return training_dataset, testing_dataset, validation_dataset
