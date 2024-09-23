import logging
from pathlib import Path

import click
import torch
import utils
from model import BasicNetwork


@click.command()
@click.option(
    "--test-data",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option("--model-path", type=click.Path(exists=True, path_type=Path))
@click.option("--batches", type=int, default=1)
@click.option("--save-dir", type=click.Path(exists=True, path_type=Path))
def main(test_data, model_path, batches, save_dir):
    # Set up logging
    logging.basicConfig(
        filename=save_dir / "training.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    model = BasicNetwork(utils.NUM_CLASSES, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path / "model.pth", weights_only=True))
    model.to(device)

    test = utils.load_datasets(test_data, batches)
    test_eval = utils.Evaluator(model, test, device)
    test_eval.run_on_input()
    test_eval.calculate_metrics()
    logger.info(f"Test evaluation: {test_eval}")


if __name__ == "__main__":
    main()
