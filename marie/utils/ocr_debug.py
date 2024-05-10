import uuid
from typing import Optional

from marie.utils.overlap import merge_bboxes_as_block
from marie.utils.utils import ensure_exists

REPLACEMENTS = {
    "$": "_DOLLAR_",
    "#": "_HASH_",
    "%": "_PERCENT_",
    "\"": "_QUOTE_",
    "'": "_SINGLE_QUOTE_",
    "(": "_OPEN_BRACKET_",
    ")": "_CLOSE_BRACKET_",
    "&": "_AND_",
    ".": "_DOT_",
    ",": "_COMA_",
    "/": "_SLASH_",
    "\\": "_SLASH_",
    ":": "_SEMI_",
    "*": "_STAR_",
    "?": "_QUESTION_",
    "<": "_SIGN_LT_",
    ">": "_SIGN_GT_",
    "|": "_PIPE_",
    "+": "_PLUS_",
    "=": "_EQUAL_",
    "-": "_MINUS_",
    "’": "_SINGLE_QUOTE_S_L_",
    "‘": "_SINGLE_QUOTE_S_R_",
    "“": "_QUOTE_SLANTED_R",
    "”": "_QUOTE_SLANTED_L",
    "–": "_DASH_SHORT_",
    "—": "_DASH_",
    "[": "_OPEN_SQUARE_BRACKET_",
    "]": "_CLOSE_SQUARE_BRACKET_",
    "@": "_SIGN_AT_",  # @
    " ": "_",
}


def normalize_label(label: str):
    return ''.join(REPLACEMENTS.get(c, c) for c in label)


def unnormalize_label(label: str):
    replacements = {v: k for k, v in REPLACEMENTS.items()}  # Reverse the dictionary
    for k, v in replacements.items():
        label = label.replace(k, v)
    return label.strip()


def dump_bboxes(
    image,
    result,
    prefix: str,
    threshold=0.90,
    text_filters: Optional[list[str]] = None,
    ngram: Optional[int] = 3,
):
    print(result)
    page_words = []
    page_boxes = []
    page_lines = []
    page_confidences = []

    for w in result["words"]:
        page_boxes.append(w["box"])
        page_words.append(w["text"])
        page_lines.append(w["line"])
        page_confidences.append(w["confidence"])

    # keep track of the ngram words to prevent mulitple detections
    ngram_keys = set()

    ngrams = [ngram - 1, ngram, ngram + 1]
    ngrams = [n for n in ngrams if 0 < n <= len(page_words)]

    for idx, text_filter in enumerate(text_filters):
        for ngram in ngrams:
            for i in range(len(page_words) - ngram + 1):
                ngram_words = page_words[i : i + ngram]
                ngram_boxes = page_boxes[i : i + ngram]
                confidence = round(sum(page_confidences[i : i + ngram]) / ngram, 4)

                if page_lines:
                    ngram_lines = page_lines[i : i + ngram]
                    if len(set(ngram_lines)) > 1:
                        print(
                            f"Skipping ngram {ngram_words} as it is not in the same line"
                        )
                        continue

                ngram_text = " ".join(ngram_words).strip().upper()
                template_text = text_filter.strip().upper()

                # check if ngram_text is contained in template_text
                # if ngram_text.startswith(template_text):
                remove_text_head = True
                # if template_text in ngram_text:
                print(
                    "compare",
                    template_text,
                    ngram_text,
                    ngram_text.startswith(template_text),
                )
                has_match = False
                if remove_text_head and ngram_text.startswith(template_text):
                    tokens = template_text.split(" ")
                    t_words = []
                    t_boxes = []

                    if len(ngram_words) > 1:
                        for b, w in zip(ngram_boxes, ngram_words):
                            print(b, w)
                            for z, t in enumerate(tokens):
                                if t != w:
                                    t_words.append(w)
                                    t_boxes.append(b)

                        has_match = True
                        ngram_boxes = t_boxes
                        ngram_words = t_words
                        ngram_text = " ".join(ngram_words).strip().upper()
                        # ngram_text = ngram_text.replace(template_text, "").strip()
                if len(ngram_text) == 0:
                    continue

                box = merge_bboxes_as_block(ngram_boxes)
                x, y, w, h = box
                # convert form xywh to xyxy
                converted = [
                    box[0],
                    box[1],
                    box[0] + box[2],
                    box[1] + box[3],
                ]
                word_img = image.crop(converted)
                label = normalize_label(ngram_text)
                root_label = f"ngram-{ngram}"

                if label in ngram_keys:
                    continue
                ngram_keys.add(label)

                if has_match or ngram_text.startswith(template_text):
                    print("Found text filter", text_filter)
                    # ensure_exists(f"/tmp/boxes/{root_label}/{label}")
                    ensure_exists(f"/tmp/boxes/{root_label}")
                    # create a unique filename to prevent overwriting using uuid
                    fname = uuid.uuid4().hex
                    name = f"{prefix}_{confidence}_{fname}"

                    word_img.save(f"/tmp/boxes/{root_label}/{name}.png")

                    with open(f"/tmp/boxes/{root_label}/{name}.txt", "w") as f:
                        f.write(ngram_text)
                        f.write(f"\n")

    return

    for i, word in enumerate(result["words"]):
        conf = word["confidence"]
        text = word["text"]

        if conf < threshold and threshold > 0.5:
            # convert form xywh to xyxy
            converted = [
                word["box"][0],
                word["box"][1],
                word["box"][0] + word["box"][2],
                word["box"][1] + word["box"][3],
            ]
            word_img = image.crop(converted)
            label = normalize_label(text)

            # check if text is only numbers
            root_label = f"alpha"
            if text.isdigit():
                root_label = f"number"

            ensure_exists(f"/tmp/boxes/{root_label}/{label}")

            with open(f"/tmp/boxes/{root_label}/{label}/label.txt", "w") as f:
                f.write(text)
                f.write(f"\n")

            # create a unique filename to prevent overwriting using uuid
            fname = uuid.uuid4().hex
            word_img.save(
                f"/tmp/boxes/{root_label}/{label}/{prefix}_{i}_{conf}_{fname}.png"
            )
