import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
from model import BasicNetwork
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

# Hardcode batch size
BATCHES = 1
DEVICE = torch.device("cuda:0")


def get_characters_from_image(img, plot_img=False, plot_img_path: Path = None):
    # Split into lines
    # I've used the row-identification idea from this StackOverflow comment
    # plus lots of manual tweaking of parameters to find something that worked for the data given here
    # https://stackoverflow.com/questions/63596796/sorting-contours-based-on-precedence-in-python-opencv/63662498#63662498
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
    morph = cv.morphologyEx(np.array(img), cv.MORPH_CLOSE, kernel)

    rowcontours, _ = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rows = []

    # Need to copy the underlying image to draw rectangles representing lines/characters
    if plot_img:
        draw_rect = ImageDraw.Draw(img.convert("RGBA"), "RGBA")
        draw_char = ImageDraw.Draw(img.convert("RGBA"), "RGBA")

    for rowcntr in rowcontours:
        xr, yr, wr, hr = cv.boundingRect(rowcntr)
        # Choose rows that have:
        # Ratio of weight/height >= 4
        # Height larger than a single character... but not too large
        if (wr / hr) >= 4 and hr > 15 and hr < 200:
            # Add a bit of flex space around the border
            borders = (xr, yr - 14, xr + wr + 10, yr + hr)
            if plot_img:
                draw_rect.rectangle(borders, outline="Red", fill=None, width=2)
            rows.append(borders)

    # Sort the rows by their y-coordinate
    rows.sort(key=lambda x: x[1])

    characters = []
    for row in rows:
        row_img = img.crop(row)
        contours, _ = cv.findContours(
            np.array(row_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        bounding_boxes = []

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w > 2 and h > 3 and w < 100:
                y_add_below = 5
                y_add_above = 10
                borders = (x, y - y_add_above, x + w, y + h + y_add_below)
                if plot_img:
                    crop_borders = (
                        x + row[0],
                        y + row[1] - y_add_above,
                        x + row[0] + w,
                        y + row[1] + h + y_add_below,
                    )
                    draw_char.rectangle(
                        crop_borders, outline="Blue", fill=None, width=2
                    )
                bounding_boxes.append(borders)
        # Sort the characters by their x-coordinate
        # This assumes each bounding box covers exactly one line of text,
        # which is not always the case
        bounding_boxes.sort(key=lambda x: x[0])
        characters.extend([row_img.crop(b) for b in bounding_boxes])

    if plot_img:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0][0].set_title("Original image")
        axs[0][1].set_title("With rectangles to separate lines")
        axs[1][0].set_title(f"Identified {len(rows)} lines")
        axs[1][1].set_title(f"Identified {len(characters)} characters")

        axs[0][0].imshow(img)
        axs[0][1].imshow(morph)
        axs[1][0].imshow(draw_rect._image)
        axs[1][1].imshow(draw_char._image)

        plt.tight_layout()

        fig.savefig(plot_img_path)
        plt.close()

    return characters


class CharacterDataset(Dataset):
    def __init__(self, char_images):
        self.char_images = char_images

    def __len__(self):
        return len(self.char_images)

    def __getitem__(self, idx):
        # Images is already loaded, so just need to convert it to a tensor
        image = pil_to_tensor(self.char_images[idx]).float()
        # Resize to 64x64
        resized_image = Resize(size=(64, 64))(image)
        # Don't return a label because we don't have any at this point in the pipeline
        # (And, as I later realized, the labels are wrong anyway!)
        return resized_image, 0


@dataclass
class Segment:
    path: Path
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    loader = None
    img = None

    def get_image_of_segment(self):
        img = Image.open(self.path).crop(
            (self.x_start, self.y_start, self.x_end, self.y_end)
        )
        self.img = img

    def get_characters_in_segment(self, img_save_dir, plot_images=False):
        char_images = get_characters_from_image(self.img, plot_images, img_save_dir)
        if len(char_images) == 0:
            raise Exception
        # Create dataloader from characters
        dataset = CharacterDataset(char_images)
        self.loader = DataLoader(dataset, batch_size=BATCHES, shuffle=False)


def read_segment_list(
    segment_list: Path, img_dir: Path, save_dir: Path, write_img: bool
):
    # Build dictionary of segments to allow quick lookup
    segments = {}

    lines = segment_list.read_text().splitlines()
    # Segment definitions are on adjacent lines, so pair them together
    # i.e. [1, 2, 3, 4, 5, 6] -> [(1, 2), (3, 4), (5, 6)]
    for start_entry, end_entry in zip(lines[::2], lines[1::2]):
        # fname, segment_num, segment_type are the same across all files
        (fname, segment_num, x_start, y_start, segment_type) = start_entry.split(";")
        (_, _, x_end, y_end, _) = end_entry.split(";")
        # We're reading the black-and-white images but the label uses grayscale
        # Remove the g.bmp at the end of the file name and replace with b.bmp
        fname_for_img = fname[:-5] + "b.bmp"
        # Convert filename + segment to the format used in the labels:
        # jn_005tg.bmp;1 -> jn_005z1
        segment_name = f"{fname[:-6]}z{segment_num}"

        # We only have labels for the TXT segments, so skip any other type (TBL, IMG, etc)
        if segment_type == "TXT":
            segment = Segment(
                img_dir / fname_for_img,
                int(x_start),
                int(y_start),
                int(x_end),
                int(y_end),
            )
            # Get the characters from the segment
            try:
                segment.get_image_of_segment()
            except FileNotFoundError:
                # This file doesn't exist, so skip it
                continue
            try:
                segment.get_characters_in_segment(
                    save_dir / (segment_name + ".png"), write_img
                )
            except Exception:
                # print("Could not find any characters in segment, skipping")
                continue
            segments[segment_name] = segment

    return segments


def predict_text_for_segment(segment_name, segment, label_dir, model):
    # Label the segment
    try:
        txt_contents = (
            (label_dir / (segment_name + ".txt"))
            .read_text(encoding="windows-1252")  # Uses Windows encoding
            .strip("\n")  # Remove trailing newlines
        )
    except FileNotFoundError:
        # No label for this text segment, so don't go further
        txt_contents = ""

    # Run model on all the characters in the segment to get the overall label text for the segment
    overall_pred = []

    for idx, data in enumerate(segment.loader, 0):
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        predicted_chars = torch.argmax(model(inputs), dim=1)
        overall_pred.append(predicted_chars)

    predicted = torch.cat(overall_pred).cpu().tolist()
    return predicted, txt_contents


def get_character_mappings(mappings_path):
    # Get mappings from character index to actual character
    char_mappings = defaultdict(lambda: "¡")  # 195 is missing a label

    with open(mappings_path, encoding="windows-1252") as f:
        for line in f.readlines():
            splt = line.split()
            if len(splt) > 1:
                idx, chr = (splt[0], splt[1])
                char_mappings[int(idx)] = chr

    return char_mappings


@click.command()
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TestSet"),
)
@click.option(
    "-d",
    "--dpi",
    multiple=False,  # The images are the same, just different DPI, so only do one at once
    type=click.Choice(["200", "300"]),
)
@click.option(
    "-t",
    "--types",
    multiple=True,
    type=click.Choice(["Book", "Journal"]),
    default=["Book", "Journal"],
)
@click.option(
    "-m", "--model_path", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option(
    "--mappings_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default="/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/Numeric/20110202-List-Code-Character-OCR-Training-Database.txt",
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    default="/scratch/gusandmich/assignment_1_bonus_q/output",
)
@click.option(
    "--img_save_path",
    type=click.Path(exists=True, path_type=Path),
    default="/scratch/gusandmich/assignment_1_bonus_q/images",
)
@click.option("--write_images", is_flag=True, default=False)
@click.option("-l", "--logging_path", type=click.Path(path_type=Path))
def main(
    input_path,
    dpi,
    types,
    model_path,
    mappings_file,
    output_path,
    img_save_path,
    write_images,
    logging_path,
):
    # Set up logging
    logging.basicConfig(
        filename=logging_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Set up the model
    model = BasicNetwork(utils.NUM_CLASSES, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # Mapping from label index to actual character
    character_mappings = get_character_mappings(mappings_file)

    for t in types:
        type_dir = input_path / t
        # Path to overall image file
        # We only read the black-and-white files
        image_dir = type_dir / "Image" / f"{dpi}dpi_BW"
        # Path to labels for each segment
        label_dir = type_dir / "Txt"
        # Path to file listing dimensions and type of each segment
        if t == "Book":
            # There's an error in the BookList where they've forgotten to label the end segment of
            # zone 2 in bt_001sg. Rather than rewriting my parsing code I've just deleted this line and
            # saved the new file to my own directory.
            segment_list = Path(
                "/home/gusandmich@GU.GU.SE/assignment_1/setup_files/BookList.txt"
            )
        else:
            segment_list = type_dir / f"{t}List.txt"
        # Split images into the labelled 'zones' and segment out the characters
        segments = read_segment_list(
            segment_list, image_dir, img_save_path, write_images
        )

        # Add labels, segment characters, and predict the characters
        for segment_name in segments:
            try:
                (pred, actual) = predict_text_for_segment(
                    segment_name, segments[segment_name], label_dir, model
                )
            except:
                # Skip if file doesn't exist for segment or no characters found in segment
                continue
            if pred is not None or actual is not None:
                pred_to_str = "".join([character_mappings[x] for x in pred])

                with open(
                    output_path / (segment_name + "_predicted.txt"),
                    "w",
                    encoding="windows-1252",
                ) as f:
                    f.write(pred_to_str)

                with open(
                    output_path / (segment_name + "_actual.txt"),
                    "w",
                    encoding="windows-1252",
                ) as f:
                    f.write(actual)


if __name__ == "__main__":
    main()
