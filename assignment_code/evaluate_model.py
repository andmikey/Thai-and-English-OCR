from pathlib import Path

import click
import torch
import utils
from model import BasicNetwork


@click.command()
@click.option("--data", type=click.Path(exists=True, path_type=Path))
@click.option("--model-path", type=click.Path(exists=True, path_type=Path))
@click.option("--batches", type=int, default=1)
@click.option("--save-dir", type=click.Path(exists=True, path_type=Path))
def main(data, model_path, batches, save_dir):
    _, test, _ = utils.load_datasets(data, batches)
    model = BasicNetwork()
    model.load_state_dict(torch.load(model_path / "model.pth", weights_only=True))

    # TODO may need to do some transforms here if different dpi, use resize: https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html
    pred_classes_test = model(test.dataset.x)
    test_eval = utils.Evaluator(pred_classes_test, test.dataset.y)
    print(f"Test evaluation: {test_eval}")
