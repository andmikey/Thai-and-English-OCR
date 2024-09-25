from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import cv2 as cv
import numpy as np
from PIL import Image


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
    characters = []

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
            # Maybe want to do some filtering here to make sure we get large enough boxes
            x, y, w, h = cv.boundingRect(cnt)
            letter_crop = self.img.crop((x, y, x + w, y + h))
            self.characters += letter_crop


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
