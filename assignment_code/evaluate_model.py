import logging
from pathlib import Path

import click
import torch
import utils
from model import BasicNetwork


@click.command()
@click.option("--test-data", type=click.Path(exists=True, path_type=Path))
@click.option("--model-path", type=click.Path(exists=True, path_type=Path))
@click.option("--batches", type=int, default=1)
@click.option("--save-dir", type=click.Path(exists=True, path_type=Path))
def main(test_data, model_path, batches, save_dir):
    test = utils.load_datasets(test_data, batches)
    model = BasicNetwork(utils.NUM_CLASSES, 64)
    model.load_state_dict(torch.load(model_path / "model.pth", weights_only=True))

    pred_classes_test = model(test.dataset.x)
    test_eval = utils.Evaluator(pred_classes_test, test.dataset.y)
    logging.info(f"Test evaluation: {test_eval}")
