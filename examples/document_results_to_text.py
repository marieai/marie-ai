import os
from pathlib import Path

import numpy as np

from marie.renderer import TextRenderer
from marie.utils.json import load_json_file
from marie.utils.utils import ensure_exists


def process_dir(src_dir: str, output_dir: str):
    """
    Processes all files in the given source directory and writes the extracted text to separate text files in the output directory.

    Args:
        src_dir (str): The path to the source directory.
        output_dir (str): The path to the output directory.

    Returns:
        None
    """
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for file_path in Path(root_asset_dir).rglob("*"):
        if not file_path.is_file():
            continue
        try:
            print(file_path)

            resolved_output_path = os.path.join(
                output_path, file_path.relative_to(root_asset_dir)
            )
            output_dir = os.path.dirname(resolved_output_path)
            filename = os.path.basename(resolved_output_path)
            name = os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)

            results = load_json_file(file_path)
            lines = results[0]["lines"]
            lines = sorted(lines, key=lambda k: k['line'])
            text_output_path = os.path.join(output_dir, f"{name}.txt")
            with open(text_output_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line["text"] + "\n")

        except Exception as e:
            print(e)


def process_dir_text_renderer(src_dir: str, output_dir: str):
    """
    Processes all files in the given source directory and writes the extracted text to separate text files in the output directory.

    Args:
        src_dir (str): The path to the source directory.
        output_dir (str): The path to the output directory.

    Returns:
        None
    """
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for file_path in Path(root_asset_dir).rglob("*"):
        if not file_path.is_file():
            continue
        try:
            print(file_path)

            resolved_output_path = os.path.join(
                output_path, file_path.relative_to(root_asset_dir)
            )
            output_dir = os.path.dirname(resolved_output_path)
            filename = os.path.basename(resolved_output_path)
            name = os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)
            text_output_path = os.path.join(output_dir, f"{name}.txt")

            results = load_json_file(file_path)
            # create a fake frames array from metadata in the results
            frames = []

            for result in results:
                meta = result["meta"]["imageSize"]
                width = meta["width"]
                height = meta["height"]
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frames.append(frame)

            renderer = TextRenderer(config={"preserve_interword_spaces": False})
            renderer.render(
                frames,
                results,
                output_file_or_dir=text_output_path,
            )

            return
        except Exception as e:
            print(e)
            return


if __name__ == "__main__":
    ensure_exists("/tmp/marie/results-text")
    if False:
        process_dir_text_renderer(
            "~/datasets/private/corr-routing/ready/annotations",
            "/tmp/marie/results-text",
        )

    if True:
        process_dir(
            "~/datasets/private/corr-routing/ready/annotations",
            "/tmp/marie/results-text",
        )
