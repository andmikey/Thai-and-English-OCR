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
    # Code here is based on this PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # Load the data
    train = utils.load_datasets(train_data, batches)
    if validation_data:
        validate = utils.load_datasets(validation_data, batches)

    # Set up the model
    model = BasicNetwork()
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)

    # Use GPU if available
    if not torch.cuda.is_available():
        print("CUDA is not available, will train on CPU")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        loss = 0.0

        for idx, data in enumerate(train.loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate performance on training and validation sets
    pred_classes_train = model(train.dataset.x)
    train_eval = utils.Evaluator(pred_classes_train, train.dataset.y)
    print(f"Train evaluation: {train_eval}")

    if validation_data:
        pred_classes_val = model(validate.dataset.x)
        val_eval = utils.Evaluator(pred_classes_val, validate.dataset.y)

    print(f"Validation evaluation: {val_eval}")

    torch.save(model.state_dict(), save_dir / "model.pth")


if __name__ == "__main__":
    main()
