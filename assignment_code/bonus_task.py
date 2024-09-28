from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def get_characters_from_image(img, plot_img=False):
    # Split into lines
    # Some ideas borrowed from this StackOverflow comment but I've done manual tweaking of parameters:
    # https://stackoverflow.com/questions/63596796/sorting-contours-based-on-precedence-in-python-opencv/63662498#63662498
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
    morph = cv.morphologyEx(np.array(img), cv.MORPH_CLOSE, kernel)

    rowcontours, _ = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rows = []

    # For drawing
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
                crop_borders = (
                    x + row[0],
                    y + row[1] - y_add_above,
                    x + row[0] + w,
                    y + row[1] + h + y_add_below,
                )
                draw_char.rectangle(crop_borders, outline="Blue", fill=None, width=2)
                bounding_boxes.append(borders)
        # Sort the characters by their x-coordinate
        bounding_boxes.sort(key=lambda x: x[0])
        characters.extend([row_img.crop(b) for b in bounding_boxes])

    if plot_img:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0][0].set_title("Original image")
        axs[0][1].set_title("With rectangles to separate lines")
        axs[1][0].set_title("Identified lines")
        axs[1][1].set_title("Identified characters")

        axs[0][0].imshow(img)
        axs[0][1].imshow(morph)
        axs[1][0].imshow(draw_rect._image)
        axs[1][1].imshow(draw_char._image)

        plt.tight_layout()

        fig.show()

    return characters


@dataclass
class Segment:
    filename: str
    path: Path
    segment_num: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    segment_type: str
    type: str
    characters = {}

    def get_image_of_segment(self):
        img = Image.open(self.path).crop(
            (self.x_start, self.y_start, self.x_end, self.y_end)
        )
        self.img = img

    def get_characters_of_segment(self):
        arr = np.array(self.img)
        # Use OpenCV to get bounding box rectangles for letters
        contours, _ = cv.findContours(arr, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Only save sufficiently large boxes
            # I found this width/height combo by manual inspection
            if w > 5 and h > 10:
                x, y, w, h = cv.boundingRect(cnt)
                # Add a bit of flex top/bottom to get vowel diacritics
                letter_crop = self.img.crop((x, y - 20, x + w, y + h + 20))
                self.characters[(x, y, x + w, y + h)] = letter_crop

        # TODO order characters left-right top-bottom


def read_segment_list(path: Path, type, img_dir):
    # Segment : segment_num : []
    segments = defaultdict(lambda: defaultdict(Segment))

    lines = path.read_text().splitlines()
    for start_entry, end_entry in zip(lines, lines[1:]):
        (fname, segment_num, x_start, y_start, segment_type) = start_entry.split(";")
        (fname, segment_num, x_end, y_end, segment_type) = end_entry.split(";")
        segment = Segment(
            fname,
            img_dir / fname,
            segment_num,
            x_start,
            y_start,
            x_end,
            y_end,
            segment_type,
            type,
        )
        segments[fname][segment_num] = segment

    for k in segments.keys():
        print(k)
        print(segments[k])
        print()


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
    multiple=True,
    # TODO we should just use the already-thresholded BW images
    # type=click.Choice(["200dpi_BW", "200dpi_Gray", "300dpi_BW", "300dpi_Gray"]),
    type=click.Choice([200, 300]),
)
@click.option(
    "-t",
    "--types",
    multiple=True,
    type=click.Choice(["Book", "Journal"]),
    default=["Book", "Journal"],
)
def main(input_path, dpi, types):
    # TODO add logic for reading + label adding
    for t in types:
        type_dir = input_path / t
        image_dir = type_dir / "Image"
        txt_dir = type_dir / "Txt"
        segment_list = type_dir / f"{t}List.txt"
        for d in dpi:
            pth = f"{d}dpi_BW"
            segments = read_segment_list(segment_list, t, image_dir)
            # TODO add txt contents, image grabber
            # TODO allow reading b/w files as well


if __name__ == "__main__":
    main()
