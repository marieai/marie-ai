from math import ceil
from os import PathLike
from typing import Any, Dict, Union

import numpy as np

from marie.renderer.renderer import ResultRenderer
from marie.utils.types import strtobool


class TextRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)

        self.logger.info(f"TextRenderer base : {self.config}")
        self.preserve_interword_spaces = False

        if "preserve_interword_spaces" in self.config:
            self.preserve_interword_spaces = strtobool(
                self.config["preserve_interword_spaces"]
            )

    @property
    def name(self):
        return "TextRenderer"

    def __render_page(
        self, image: np.ndarray, result: Dict[str, Any], page_index: int
    ) -> str:
        """Render single result page into text"""
        # 8px X 22px = 2.75 pytorch
        # 8px X 19px = 2.375 vscode
        # https://lists.w3.org/Archives/Public/w3c-wai-gl/2017AprJun/0951.html

        if image is None:
            raise Exception("Image or list of images expected")

        self.check_format_xywh(result, True)

        # Reshape document for better on screen rendering
        # 1280×720
        # 1280×1080

        shape = image.shape

        h = shape[0]
        w = shape[1]
        char_ratio = 2.75
        char_width = 8.44  # TODO : This needs to be supplied from the box processor
        char_height = 16  # int(char_width * char_ratio)
        cols = ceil(w // char_width)
        rows = ceil(h // char_height)

        x_space = np.arange(0, w, 1)
        bins = np.linspace(0, w, cols)
        bins = np.array(bins).astype(np.int32)
        x_hist = np.digitize(x_space, bins, right=True)

        # ['meta', 'words', 'lines']
        if True:
            print(f"Image size  : {shape}")
            print(f"Char ratio  : {char_ratio}")
            print(f"Char width  : {char_width}")
            print(f"Char height : {char_height}")
            print(f"Columns     : {cols}")
            print(f"Rows        : {rows}")
            print(f"bins        : {bins}")
            print(f"x_hist      : {x_hist}")

        meta = result["meta"]
        words = result["words"]
        lines = result["lines"]

        buffer = ""
        start_cell_y = 1
        force_word_index_sort = False
        min_spacing = 500
        max_characters_per_line = ceil(w // char_width)

        print(f"max_characters_per_line = {max_characters_per_line}")

        for i, line in enumerate(lines):
            bbox = line["bbox"]
            # this are Word ID not Indexes, each word is assigned a unique ID when it is created
            wordids = line["wordids"]
            x, y, w, h = bbox
            baseline = y + h
            cell_y = baseline // char_height
            delta_cell_y = cell_y - start_cell_y
            start_cell_y = cell_y

            if False:
                print(
                    f"Baseline # {i} : {baseline}, cell-y = {cell_y} , delta_cell_y = {delta_cell_y}"
                )

            for j in range(1, delta_cell_y):
                buffer += "\n"

            aligned_words = [w for w in words if w["id"] in wordids]

            if True or force_word_index_sort:
                word_index_picks = []
                word_picks = []
                for word in aligned_words:
                    word_index_picks.append(word["word_index"])
                    word_picks.append(word)

                word_index_picks = np.array(word_index_picks)
                word_picks = np.array(word_picks)
                sort_index = np.argsort(word_index_picks)
                aligned_words = word_picks[sort_index]

            last_space = 0
            line_buffer = " " * max_characters_per_line

            for idx, word in enumerate(aligned_words):
                # estimate space gap
                spaces = 0
                curr_box = aligned_words[idx]["box"]
                text = word["text"]
                x2, y2, w2, h2 = curr_box

                grid_space = x_hist[x2]

                spaces = grid_space - last_space
                last_space = grid_space

                # if spaces < min_spacing:
                #     min_spacing = spaces

                line_buffer = line_buffer[:grid_space] + text + line_buffer[grid_space:]
                print(f"{grid_space} : {spaces}  > {text}")

            print(line_buffer)
            buffer += line_buffer
            if i < len(lines) - 1:
                buffer += "\n"

        # buffer = buffer.replace(" " * 8, " ")
        # print(f"min_spacing = {min_spacing}")
        return buffer

    def render(
        self,
        frames: np.ndarray,
        results: [Dict[str, Any]],
        output_filename: Union[str, PathLike],
    ) -> None:
        """Renders results into text output stream
        Results parameter "format" is expected to be in "XYWH" conversion will be performed to accommodate this
        """
        self.logger.info(f"Render textfile : {output_filename}")
        # The form feed character is sometimes used in plain text files of source code as a delimiter for a page break
        page_seperator = "\f"  # or \x0c
        # page_seperator = "\n\n__SEP__\n\n"
        buffer = ""
        # page_index is not same as page_number, page_index is an index into the array however frames can be assembled
        # in our of order or can be picked
        for page_index, (image, result) in enumerate(zip(frames, results)):
            try:
                content = self.__render_page(image, result, page_index)
                buffer += content
            except Exception as e:
                self.logger.error(e, stack_info=True, exc_info=True)

            if len(frames) > 1:
                if page_index < len(frames) - 1:
                    buffer += page_seperator

        with open(output_filename, "w", encoding="UTF-8") as text_file:
            text_file.write(buffer)
