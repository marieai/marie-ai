import os
import traceback
from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
from PIL import Image
from collections import defaultdict

from marie.boxes import PSMode
from marie.constants import __model_path__
from marie.document import TrOcrProcessor
from marie.document.craft_ocr_processor import CraftOcrProcessor
from marie.document.lev_ocr_processor import LevenshteinOcrProcessor
from marie.document.tesseract_ocr_processor import TesseractOcrProcessor
from marie.ocr import OcrEngine, CoordinateFormat
from collections import OrderedDict

from marie.utils.json import store_json_object


class VotingOcrEngine(OcrEngine):
    """
    An implementation of OcrEngine which includes voting.

    Default processors: TrOcrIcrProcessor, CraftIcrProcessor, TesseractOcrProcessor

    Notes :
    https://www.atalasoft.com/docs/dotimage/docs/html/N_Atalasoft_Ocr_Voting.htm
    https://github.com/HasithaSuneth/Py-Tess-OCR/blob/main/Py-Tess-OCR%20(Linux).py
    """

    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(models_dir=models_dir, cuda=cuda, **kwargs)

        self.processors = OrderedDict()

        self.processors["trocr"] = {
            "enabled": True,
            "default": True,
            "processor": TrOcrProcessor(work_dir=self.work_dir_icr, cuda=self.has_cuda),
        }

        self.processors["craft"] = {
            "enabled": True,
            "processor": CraftOcrProcessor(
                work_dir=self.work_dir_icr, cuda=self.has_cuda
            ),
        }

        # Good results but it is missing some characters from special set (e.g. `/ \ ! ? : ;`) etc
        # Before enabling this, we need to fine tune the model
        self.processors["levocr"] = {
            "enabled": False,
            "processor": LevenshteinOcrProcessor(
                work_dir=self.work_dir_icr, cuda=self.has_cuda
            ),
        }

        self.processors["tesseract"] = {
            "enabled": True,
            "processor": TesseractOcrProcessor(
                work_dir=self.work_dir_icr, cuda=self.has_cuda
            ),
        }

    def extract(
        self,
        frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYXY,
        regions: [] = None,
        queue_id: str = None,
        **kwargs,
    ):
        is_default = False
        default_results = None
        aggregated_results = OrderedDict()
        for key, val in self.processors.items():
            try:
                print(f"Processing with {key}")
                if val["enabled"]:
                    if "default" in val and val["default"]:
                        is_default = True
                    icr_processor = val["processor"]
                    self.logger.info(
                        f"Processing with {icr_processor.__class__.__name__}"
                    )
                    results = self.process_single(
                        self.box_processor,
                        icr_processor,
                        frames,
                        pms_mode,
                        coordinate_format,
                        regions,
                        queue_id,
                        **kwargs,
                    )
                    aggregated_results[key] = results
                    if is_default:
                        default_results = results
            except Exception as e:
                traceback.print_exc()
                continue

        return self.voting_evaluator(aggregated_results, default_results)

    def voting_evaluator(
        self, aggregated_results: OrderedDict[str, List], default_results
    ) -> List[Dict]:
        """
        Evaluate the results and return the best result

        :param aggregated_results:
        :param default_results:
        :return:
        """
        self.logger.info("Voting evaluator")
        candidate_words_by_page = {}
        for key, results in aggregated_results.items():
            print(f"Result processor : {key}")
            for idx, page_result in enumerate(results):
                if idx not in candidate_words_by_page:
                    candidate_words_by_page[idx] = {}
                for word in page_result["words"]:
                    word_id = word["id"]
                    word["processor"] = key
                    if word_id not in candidate_words_by_page[idx]:
                        candidate_words_by_page[idx][word_id] = []
                    candidate_words_by_page[idx][word_id].append(word)
                    print(
                        f"candidate_words[idx][word_id] = {len(candidate_words_by_page[idx][word_id])}  > {idx}  {word_id}"
                    )

        # pick the word with the highest confidence
        words_by_confidence_by_page = {}

        for key, candidates in candidate_words_by_page.items():
            idx = key
            self.logger.info(f"Word : {idx}")
            print(f"Word by page: {idx}")
            if idx not in words_by_confidence_by_page:
                words_by_confidence_by_page[idx] = []

            for word_id, words in candidates.items():
                print(f"Word : {word_id}")
                word = words[0]
                if len(words) > 1:
                    # pick the word with the highest confidence
                    for word_candidate in words:
                        if word_candidate["confidence"] > word["confidence"]:
                            word = word_candidate

                words_by_confidence_by_page[idx].append(word)

        min_vote_count = 2
        words_by_vote_by_page = {}

        for key, candidates in candidate_words_by_page.items():
            idx = key
            self.logger.info(f"Word : {idx}")
            print(f"Word by page: {idx}")
            if idx not in words_by_vote_by_page:
                words_by_vote_by_page[idx] = []

            for word_id, words in candidates.items():
                self.logger.info(f"Word : {word_id}")
                print(f"Word : {word_id}")
                groups = defaultdict(list)
                for word in words:
                    groups[word["text"]].append(word)

                # pick the group with the highest vote count
                best_group = []
                for group in groups.values():
                    if len(group) >= len(best_group):
                        if len(group) == len(best_group):
                            group_conf = sum([word["confidence"] for word in group])
                            best_group_conf = sum(
                                [word["confidence"] for word in best_group]
                            )
                            if group_conf > best_group_conf:
                                best_group = group
                        else:
                            best_group = group

                if len(best_group) >= min_vote_count:
                    selected_word = best_group[0]
                    selected_word["strategy"] = {
                        "type": "voting",
                        "candidates": len(best_group),
                        "votes": deepcopy(best_group),
                    }
                    for vote_word in selected_word["strategy"]["votes"]:
                        vote_word.pop("box", None)
                        vote_word.pop("strategy", None)
                else:
                    # default to the first word as this is the default processor
                    selected_word = words[0]
                    selected_word["strategy"] = {"type": "default"}
                    for word in words_by_confidence_by_page[idx]:
                        if word["id"] == word_id:
                            if word["confidence"] > selected_word["confidence"]:
                                selected_word = word
                                selected_word["strategy"] = {
                                    "type": "confidence",
                                    "confidence": word["confidence"],
                                }
                                break
                words_by_vote_by_page[idx].append(selected_word)

        print(f"words_by_vote_by_page : {len(words_by_vote_by_page)}")
        store_json_object(
            words_by_vote_by_page, "/tmp/marie/words_by_vote_by_page.json"
        )

        # pick the words by page and update the results
        output_results = deepcopy(default_results)
        for idx, page_result in enumerate(output_results):
            page_result["words"] = words_by_vote_by_page[idx]

        return output_results
