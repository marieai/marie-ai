from math import ceil

import numpy as np

from renderer.renderer import ResultRenderer
from utils.types import strtobool


class TextRenderer(ResultRenderer):
    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        print(f"TextRenderer base : {config}")

        self.preserve_interword_spaces = False
        if "preserve_interword_spaces" in config:
            self.preserve_interword_spaces = strtobool(config["preserve_interword_spaces"])

    @property
    def name(self):
        return "TextRenderer"

    def render(self, image, result, output_filename):
        print("Rendering ...")
        # 8px X 22px = 2.75 pytorch
        # 8px X 19px = 2.375 vscode

        char_ratio = 2.75
        char_width = 20  # 8
        char_height = int(char_width * char_ratio)
        shape = image.shape

        print(f"Char ratio : {char_ratio}")
        print(f"Char width : {char_width}")
        print(f"Char height : {char_height}")
        print(f"Image size : {shape}")

        h = shape[0]
        w = shape[1]

        xs = ceil(h / char_width)
        hs = ceil(w / char_height)
        bins = hs * xs
        print(f"Segments size [hs, xs, bins]: {hs},  {xs}, {bins}")
        # ['meta', 'words', 'lines']
        meta = result["meta"]
        words = result["words"]
        lines = result["lines"]

        for i, line in enumerate(lines):
            print(line)
            bbox = line["bbox"]
            wordids = line["wordids"]
            x, y, w, h = bbox
            baseline = y + h
            celly = baseline // char_height
            print(f"Baseline # {i} : {baseline}, cell-y = {celly}")

            # we need to sort the words id their 'x'  as the wordids can be out of order.
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
            print(sort_index)
            aligned_words = word_picks[sort_index]

            # def add_column(val, col_len)->str:
            print("Aligned")
            buffer = ""

            # TODO : This needs to be supplied from the box processor
            estimate_character_width = 26
            print(f"self.preserve_interword_spaces = {self.preserve_interword_spaces}")

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

                print(f"gap :  {idx} : >  {gap}, spaces = {spaces}")

                text = word["text"]
                confidence = word["confidence"]
                box = word["box"]
                x, y, w, h = box
                cellx = x // char_width
                cols = (x + w) // char_width

                print(f"{cellx}, {cols} :: {celly}     >>   {box} :: {text}")
                buffer += " " * spaces
                buffer += text
                # print(f'buffer : {buffer}')

            print("Final ----")
            print(buffer)
