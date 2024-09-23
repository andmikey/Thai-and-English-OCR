from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click


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

    def get_image_of_segment(self):
        pass


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
    "-c",
    "--category",
    multiple=True,
    type=click.Choice(["200dpi_BW", "200dpi_Gray", "300dpi_BW", "300dpi_Gray"]),
)
@click.option(
    "-t",
    "--types",
    multiple=True,
    type=click.Choice(["Book", "Journal"]),
    default=["Book", "Journal"],
)
def main(input_path, category, types):
    for t in types:
        type_dir = input_path / t
        image_dir = type_dir / "Image"
        txt_dir = type_dir / "Txt"
        segment_list = type_dir / f"{t}List.txt"
        segments = read_segment_list(segment_list, t, image_dir)
        # TODO add txt contents, image grabber


if __name__ == "__main__":
    main()
