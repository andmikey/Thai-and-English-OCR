import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from model import BasicNetwork


@click.command()
@click.option(
    "--train-data",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
)
@click.option(
    "--validation-data",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=False,
)
@click.option("--batches", type=int, default=1)
@click.option("--epochs", type=int, default=100)
@click.option("--save-dir", type=click.Path(exists=True, path_type=Path))
def main(
    train_data,
    validation_data,
    batches,
    epochs,
    save_dir,
):
    # Set up logging
    logging.basicConfig(
        filename=save_dir / "training.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load the data
    train = utils.load_datasets(train_data, batches)
    if validation_data:
        validate = utils.load_datasets(validation_data, batches)

    # Set up the model
    model = BasicNetwork(utils.NUM_CLASSES, 64)
    # Define loss function and optimizer
    # Optimizer takes class index
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    # Use GPU if available
    if not torch.cuda.is_available():
        print("CUDA is not available, will train on CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for idx, data in enumerate(train.loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate performance on training and validation sets
    with torch.no_grad():
        # Evaluate on train
        train_eval = utils.Evaluator(model, train, device)
        train_eval.run_on_input()
        train_eval.calculate_metrics()
        logger.info(f"Training evaluation: {train_eval}")

        # Evaluate on validation
        val_eval = utils.Evaluator(model, validate, device)
        val_eval.run_on_input()
        val_eval.calculate_metrics()
        logger.info(f"Validation evaluation: {val_eval}")

    logger.info(f"Saving model to {save_dir / 'model.pth'}")
    torch.save(model.state_dict(), save_dir / "model.pth")


if __name__ == "__main__":
    main()
