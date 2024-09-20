import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import click


@dataclass
class TrainingDataPoint:
    language: str
    dpi: str
    style: str
    path: Path
    class_name: str

    def __str__(self):
        return ",".join(
            [self.language, self.dpi, self.style, self.class_name, str(self.path)]
        )


class TrainingDataSet:

    def __init__(self):
        self.data_points = defaultdict(list)

    def add_points(self, key: Set[str], points: List[TrainingDataPoint]):
        self.data_points[key].extend(points)

    def write_to_file(self, file_path: Path, file_name: Path):
        with open(file_path / file_name, "w+") as f:
            for key in self.data_points.keys():
                for point in self.data_points[key]:
                    f.write(f"{point}\n")

    def count_points(self):
        total = 0
        for key in self.data_points.keys():
            total += len(self.data_points[key])

        return total


def split_dataset(
    items: List,
    train_proportion: float,
    test_proportion: float,
    validation_proportion: float,
) -> Tuple[List, List, List]:
    # Shuffle the list before doing anything to it
    random.shuffle(items)

    num_train = int(train_proportion * len(items))
    num_test = int(test_proportion * len(items))

    train = items[:num_train]
    test = items[num_train : (num_train + num_test)]
    val = items[(num_train + num_test) :]

    return train, test, val


@click.command()
@click.option(
    "-l",
    "--language",
    multiple=True,
    default=["English", "Thai"],
    type=click.Choice(["English", "Thai", "Special", "Numeric"]),
)  # Empty means use all
@click.option(
    "-d",
    "--dpi",
    multiple=True,
    default=["200", "300", "400"],
    type=click.Choice(["200", "300", "400"]),
)
@click.option(
    "-s",
    "--style",
    multiple=True,
    default=["bold", "bold_italic", "italic", "normal"],
    type=click.Choice(["bold", "bold_italic", "italic", "normal"]),
)
@click.option("-trp", "--train_proportion", default=0.6, type=click.FloatRange(0, 1))
@click.option("-tep", "--test_proportion", default=0.2, type=click.FloatRange(0, 1))
@click.option(
    "-vap", "--validation_proportion", default=0.2, type=click.FloatRange(0, 1)
)
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet"),
)
@click.option("-o", "--output_path", type=click.Path(exists=True, path_type=Path))
@click.option("-r", "--random_seed", type=int, default=42, required=False)
def main(
    language,
    dpi,
    style,
    train_proportion,
    test_proportion,
    validation_proportion,
    input_path,
    output_path,
    random_seed,
):
    # Validate proportion inputs
    if train_proportion + test_proportion + validation_proportion != 1:
        raise AssertionError(
            "Train, test, and validation proportions should sum to one but summed to"
            + f"{train_proportion + test_proportion + validation_proportion}"
        )

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Holders of train/test/val datasets
    training_set = TrainingDataSet()
    testing_set = TrainingDataSet()
    validation_set = TrainingDataSet()

    # Go through all languages
    for lang in language:
        lang_dir = input_path / lang
        # Get all the character directories
        character_dirs = [f for f in lang_dir.iterdir() if f.is_dir()]
        for character_dir in character_dirs:
            char_name = character_dir.parts[-1]
            # Go through all dpis
            for dpi_val in dpi:
                # Go through all styles
                for style_val in style:
                    # Get all the image files and split them into train/test
                    images = [
                        TrainingDataPoint(lang, dpi_val, style_val, f, char_name)
                        for f in (character_dir / dpi_val / style_val).iterdir()
                        if f.is_file()
                    ]

                    train, test, val = split_dataset(
                        images, train_proportion, test_proportion, validation_proportion
                    )

                    key = (lang, dpi, style, char_name)

                    training_set.add_points(key, train)
                    testing_set.add_points(key, test)
                    validation_set.add_points(key, val)

    # Write out the train/test/validation sets to the given output path
    print(
        f"Generated data points:\n Train: {training_set.count_points()}\n",
        f"Test:  {testing_set.count_points()}\n",
        f"Val:   {validation_set.count_points()}\n",
    )
    training_set.write_to_file(output_path, "training_set.txt")
    testing_set.write_to_file(output_path, "testing_set.txt")
    validation_set.write_to_file(output_path, "validation_set.txt")


if __name__ == "__main__":
    main()
