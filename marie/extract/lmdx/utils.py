import re


def format_segment(row):
    a_formatted = f'{row["x-mid"]:02d}'
    b_formatted = f'{row["y-mid"]:02d}'
    return f"{a_formatted}|{b_formatted}"


def regex_split(pattern, string):
    splits = []
    last_index = 0
    for match in re.finditer(pattern, string):
        splits.append(string[last_index : match.start()])
        splits.append(match.group())
        last_index = match.end()
    splits.append(string[last_index:])
    return splits


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]
