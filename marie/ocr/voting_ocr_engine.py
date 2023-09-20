import os
import traceback
from typing import List, Union, Dict

import numpy as np
from PIL import Image
from collections import defaultdict

from marie.boxes import PSMode
from marie.constants import __model_path__
from marie.document import TrOcrIcrProcessor
from marie.document.craft_icr_processor import CraftIcrProcessor
from marie.document.tesseract_icr_processor import TesseractOcrProcessor
from marie.ocr import OcrEngine, CoordinateFormat
from collections import OrderedDict


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
            "processor": TrOcrIcrProcessor(
                work_dir=self.work_dir_icr, cuda=self.has_cuda
            ),
        }
        self.processors["craft"] = {
            "enabled": True,
            "processor": CraftIcrProcessor(
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
        aggregate_results = []
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
                    print(results)
                    aggregate_results.append({"processor": key, "results": results})
                    if is_default:
                        default_results = results
            except Exception as e:
                traceback.print_exc()
                continue

        return self.voting_evaluator(aggregate_results, default_results)

    def voting_evaluator(
        self, results: Dict[List[Dict]], default_results
    ) -> List[Dict]:
        """
        Evaluate the results and return the best result

        :param results:
        :param default_results:
        :return:
        """
        self.logger.info("Voting evaluator")
        # map each result to a dictionary of key and value
        candidate_words = {}
        for key, value in results.items():
            processor = value["processor"]
            result = value["results"]
            self.logger.info(f"Result : {key}")
            print(f"Result : {key}")
            for word in result["words"]:
                word_id = word["id"]
                if word_id not in candidate_words:
                    candidate_words[word_id] = []
                word["processor"] = processor
                candidate_words[word_id].append(word)

        # pick the word with the highest confidence
        words_by_confidence = []
        for word_id, words in candidate_words.items():
            self.logger.info(f"Word : {word_id}")
            print(f"Word : {word_id}")
            word = words[0]
            if len(words) > 1:
                # pick the word with the highest confidence
                for word_candidate in words:
                    if word_candidate["confidence"] > word["confidence"]:
                        word = word_candidate
            words_by_confidence.append(word)

        words_by_vote = []
        min_vote_count = 2

        for word_id, words in candidate_words.items():
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
            else:
                # default to the first word as this is the default processor
                selected_word = words[0]
                for word in words_by_confidence:
                    if word["id"] == word_id:
                        selected_word = word
                        break
            words_by_vote.append(selected_word)
        return []
