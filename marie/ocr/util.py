def get_words_and_boxes(ocr_results, page_index: int) -> tuple:
    """
    Get words and boxes from OCR results.
    :param ocr_results:
    :param page_index:
    :return:
    """
    words = []
    boxes = []

    if not ocr_results:
        return words, boxes

    if page_index >= len(ocr_results):
        raise ValueError(f"Page index {page_index} is out of range.")

    for w in ocr_results[page_index]["words"]:
        boxes.append(w["box"])
        words.append(w["text"])
    return words, boxes
