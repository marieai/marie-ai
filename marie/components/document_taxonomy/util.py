import copy
from typing import Dict, List

from marie.components.document_taxonomy.datamodel import TaxonomyPrediction


def merge_annotations(
    annotations: List[TaxonomyPrediction], src_metadata: dict, key: str = "taxonomy"
) -> dict:
    metadata = copy.deepcopy(src_metadata)
    for annotation in annotations:
        for line in metadata["lines"]:
            if line["line"] == annotation.line_id:
                line[key] = {"label": annotation.label, "score": annotation.score}
                break
    print(metadata)
    return metadata


def group_taxonomies_by_label(lines: List[Dict], key: str) -> List[Dict]:
    """
    Groups contiguous lines with the same label into taxonomy groups.
    """
    if len(lines) == 0:
        return []

    grouped_lines = []
    current_group = {"label": lines[0][key]["label"], "lines": [lines[0]]}

    for line in lines[1:]:
        if line[key]["label"] == current_group["label"]:
            current_group["lines"].append(line)
        else:
            grouped_lines.append(current_group)
            current_group = {"label": line[key]["label"], "lines": [line]}

    grouped_lines.append(current_group)  # Add the last group

    for group in grouped_lines:
        print(f"Group: {group['label']}")
        group_size = len(group["lines"])
        total_score = 0
        min_x, min_y, max_x, max_y = (
            float('inf'),
            float('inf'),
            float('-inf'),
            float('-inf'),
        )
        for line in group["lines"]:
            score = line[key]['score']
            total_score += score
            score = f"{score:.4f}"
            line_info = f"Line {line['line']}: {score} > {line['text']}"
            print(line_info)
            bbox = line['bbox']
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            max_x = max(max_x, bbox[0] + bbox[2])
            max_y = max(max_y, bbox[1] + bbox[3])
        average_score = total_score / group_size
        print(f"Average Score for Group '{group['label']}': {average_score:.4f}")
        group['bbox'] = [min_x, min_y, max_x - min_x, max_y - min_y]
        group['score'] = average_score
        print(f"Bounding Box for Group '{group['label']}': {group['bbox']}")
    return grouped_lines
