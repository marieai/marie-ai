import uuid

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


def dump_bboxes(image, result, prefix: str, threshold=0.90):
    # dump all the words to a directory for debugging if the confidence is below a certain threshold
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
