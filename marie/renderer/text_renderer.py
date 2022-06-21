from math import ceil

import numpy as np

from marie.renderer.renderer import ResultRenderer
from marie.utils.types import strtobool


class TextRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"TextRenderer base : {config}")

        self.preserve_interword_spaces = False
        if "preserve_interword_spaces" in config:
            self.preserve_interword_spaces = strtobool(
                config["preserve_interword_spaces"]
            )

    @property
    def name(self):
        return "TextRenderer"

    def __render_page(self, image, result, page_index):
        """Render page into text"""
        # 8px X 22px = 2.75 pytorch
        # 8px X 19px = 2.375 vscode
        if image is None:
            raise Exception("Image or list of images expected")

        char_ratio = 2.75
        char_width = 20  # 8
        char_height = int(char_width * char_ratio)
        shape = image.shape

        # print(f"Char ratio : {char_ratio}")
        # print(f"Char width : {char_width}")
        # print(f"Char height : {char_height}")
        # print(f"Image size : {shape}")

        h = shape[0]
        w = shape[1]

        xs = ceil(h / char_width)
        hs = ceil(w / char_height)
        # ['meta', 'words', 'lines']
        meta = result["meta"]
        words = result["words"]
        lines = result["lines"]

        buffer = ""
        start_cell_y = 1

        for i, line in enumerate(lines):
            bbox = line["bbox"]
            wordids = line["wordids"]
            x, y, w, h = bbox
            baseline = y + h
            cell_y = baseline // char_height
            delta_cell_y = cell_y - start_cell_y
            start_cell_y = cell_y

            # print(
            #     f"Baseline # {i} : {baseline}, cell-y = {cell_y} , delta_cell_y = {delta_cell_y}"
            # )

            for j in range(1, delta_cell_y):
                buffer += "\n"
            # we need to sort the words id their 'x'  as the wordids can be out of order
            word_ids = []
            box_picks = []
            word_picks = []

            for wid in wordids:
                word = words[wid]
                word_ids.append(word["id"])
                box_picks.append(word["box"])
                word_picks.append(word)

            box_picks = np.array(box_picks)
            word_picks = np.array(word_picks)

            x1 = box_picks[:, 0]
            sort_index = np.argsort(x1)
            aligned_words = word_picks[sort_index]

            # TODO : This needs to be supplied from the box processor
            estimate_character_width = 26

            for idx, word in enumerate(aligned_words):
                # estimate space gap
                spaces = 0
                curr_box = aligned_words[idx]["box"]
                x2, y2, w2, h2 = curr_box
                if idx > 0:
                    prev_box = aligned_words[idx - 1]["box"]
                    x1, y1, w1, h1 = prev_box
                    gap = abs(x1 + w1 - x2)
                    spaces = 1
                else:
                    gap = x2

                if self.preserve_interword_spaces:
                    if gap > estimate_character_width:
                        spaces = max(1, gap // estimate_character_width)

                # print(f"gap :  {idx} : >  {gap}, spaces = {spaces}")

                text = word["text"]
                confidence = word["confidence"]
                box = word["box"]
                x, y, w, h = box
                cellx = x // char_width
                cols = (x + w) // char_width

                # print(f"{cellx}, {cols} :: {cell_y}     >>   {box} :: {text}")
                buffer += " " * spaces
                buffer += text

            if i < len(lines) - 1:
                buffer += "\n"

        return buffer

    def render(self, frames, results, output_filename):
        """Renders results into output stream"""
        # The form feed character is sometimes used in plain text files of source code as a delimiter for a page break
        page_seperator = "\f"  # or \x0c
        # page_seperator = "\n\n__SEP__\n\n"
        buffer = ""
        for page_index, (image, result) in enumerate(zip(frames, results)):
            content = self.__render_page(image, result, page_index)
            buffer += content
            if len(frames) > 1:
                if page_index < len(frames) - 1:
                    buffer += page_seperator

        print("------- Final ------")
        print(buffer)

        with open(output_filename, "w", encoding="UTF-8") as text_file:
            text_file.write(buffer)

        return buffer
