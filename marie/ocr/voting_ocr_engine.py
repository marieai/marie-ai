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
                if val["enabled"]:
                    if "default" in val and val["default"]:
                        is_default = True
                    icr_processor = val["processor"]
                    self.logger.debug(
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
                    # store_json_object(results, f"/tmp/marie/results_{key}.json")
            except Exception as e:
                traceback.print_exc()
                continue

        return self.voting_evaluator(aggregated_results, default_results, regions)

    def group_candidates_by_selector(
        self, aggregated_results: OrderedDict[str, List], selector: str = None
    ) -> Dict:
        candidate_words_by_selector = {}
        for key, aggregated_result in aggregated_results.items():
            # print(f"Result processor : {key}, selector : {selector}")
            if selector is not None:
                results = aggregated_result[selector]  # extended
            else:
                results = aggregated_result

            for idx, page_result in enumerate(results):
                if selector is not None and selector == "extended":
                    idx = page_result["id"]
                if idx not in candidate_words_by_selector:
                    candidate_words_by_selector[idx] = {}
                if "words" not in page_result:
                    self.logger.warning(f"Page has no words : {idx}")
                    continue
                for word in page_result["words"]:
                    word_id = word["id"]
                    word["processor"] = key
                    if word_id not in candidate_words_by_selector[idx]:
                        candidate_words_by_selector[idx][word_id] = []
                    candidate_words_by_selector[idx][word_id].append(word)
                    # print(
                    #     f"candidate_words_by_selector[idx][word_id] = {len(candidate_words_by_selector[idx][word_id])}  > {idx}  {word_id}"
                    # )

        return candidate_words_by_selector

    def get_words_by_confidence_by_selector(
        self, candidate_words_by_selector: Dict
    ) -> Dict:
        """
         pick the word with the highest confidence
        :param candidate_words_by_selector:
        :return:
        """
        words_by_confidence_by_selector = {}

        for key, candidates in candidate_words_by_selector.items():
            idx = key
            self.logger.debug(f"Word : {idx}")
            if idx not in words_by_confidence_by_selector:
                words_by_confidence_by_selector[idx] = []

            for word_id, words in candidates.items():
                word = words[0]
                if True or len(words) > 1:
                    # pick the word with the highest confidence
                    for word_candidate in words:
                        if word_candidate["confidence"] > word["confidence"]:
                            word = word_candidate

                words_by_confidence_by_selector[idx].append(word)
                # print(
                #     f"words_by_confidence_by_selector[idx][word_id] = {words_by_confidence_by_selector[idx][word_id]}  > {idx}  {word_id}"
                # )
        return words_by_confidence_by_selector

    def get_words_by_vote_by_selector(
        self, candidate_words_by_selector: Dict, min_vote_count: int = 2
    ) -> Dict:
        """
        pick the word with the highest confidence
        :param self:
        :param candidate_words_by_selector:
        :param min_vote_count:
        :return:
        """
        words_by_vote_by_selector = {}

        for key, candidates in candidate_words_by_selector.items():
            idx = key
            if idx not in words_by_vote_by_selector:
                words_by_vote_by_selector[idx] = []

            for word_id, words in candidates.items():
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
                    # print(f"IDX to check : {idx}")
                    # print(f"word_id to check : {word_id}")
                    selected_word = words[0]
                    selected_word["strategy"] = {"type": "default"}

                    for word in candidate_words_by_selector[idx][word_id]:
                        if word["id"] == word_id:
                            if word["confidence"] > selected_word["confidence"]:
                                selected_word = word
                                selected_word["strategy"] = {
                                    "type": "confidence",
                                    "confidence": word["confidence"],
                                }
                                break
                words_by_vote_by_selector[idx].append(selected_word)

        store_json_object(
            words_by_vote_by_selector, "/tmp/marie/words_by_vote_by_selector.json"
        )

        return words_by_vote_by_selector

    def voting_evaluator(
        self,
        aggregated_results: OrderedDict[str, List],
        default_results,
        regions: [] = None,
    ) -> List[Dict]:
        """
        Evaluate the results and return the best result

        :param aggregated_results: The results from all the processors
        :param default_results: The results from the default processor
        :param regions: The regions to evaluate (if any)
        :return: The best result from the evaluation
        """

        self.logger.info("Voting evaluator")
        has_regions = regions is not None and len(regions) > 0
        debug_results = False

        if has_regions:
            # check if we have any results to evaluate
            if len(aggregated_results) == 0:
                self.logger.warning("No results to evaluate")

                # create a default result set with confidence 0 for each region
                if default_results is None:
                    output_results = {}
                else:
                    output_results = deepcopy(default_results)

                output_results["regions"] = []
                for region in regions:
                    region["confidence"] = 0
                    region["text"] = ""
                    region["original_text"] = ""
                    region["words"] = []
                    output_results["regions"].append(region)
                return output_results

            candidate_words_by_selector = self.group_candidates_by_selector(
                aggregated_results, "extended"
            )

            if debug_results:
                store_json_object(
                    candidate_words_by_selector, "/tmp/marie/region_step-1.json"
                )

            # pick the word with the highest confidence
            words_by_confidence_by_selector = self.get_words_by_confidence_by_selector(
                candidate_words_by_selector
            )

            if debug_results:
                store_json_object(
                    words_by_confidence_by_selector, "/tmp/marie/region_step-2.json"
                )

            words_by_vote_by_selector = self.get_words_by_vote_by_selector(
                candidate_words_by_selector, 2
            )

            if debug_results:
                store_json_object(
                    words_by_vote_by_selector, "/tmp/marie/region_step-3.json"
                )

            output_results = deepcopy(default_results)

            if debug_results:
                store_json_object(
                    output_results, "/tmp/marie/region_output_results.json"
                )

            # pick the words by page and update the results
            extended = output_results["extended"]
            for selector_key, selector_value in words_by_vote_by_selector.items():
                for idx, extended_result in enumerate(extended):
                    if extended_result["id"] == selector_key:
                        extended_result["words"] = words_by_vote_by_selector[
                            selector_key
                        ]

            # by forcing the type to string, we can avoid the issue with DataRequest parameters(self) -> Dict:
            # update the regions
            for region in output_results["regions"]:
                region_id = region["id"] = str(region["id"])
                for idx, extended_result in enumerate(extended):
                    if extended_result["id"] == region_id:
                        extended_result["words"].sort(key=lambda x: x["word_index"])
                        text = " ".join(
                            [word["text"] for word in extended_result["words"]]
                        )

                        confidence = 0
                        if len(extended_result["words"]) > 0:
                            confidence = sum(
                                [
                                    word["confidence"]
                                    for word in extended_result["words"]
                                ]
                            ) / len(extended_result["words"])

                        region["original_text"] = region["text"]
                        region["text"] = text
                        region["confidence"] = round(confidence, 4)

            store_json_object(
                output_results, "/tmp/marie/region_output_results-moded.json"
            )

            return output_results

        else:
            candidate_words_by_page = {}
            for key, results in aggregated_results.items():
                print(f"Result processor : {key}")
                for idx, page_result in enumerate(results):
                    if idx not in candidate_words_by_page:
                        candidate_words_by_page[idx] = {}
                    if "words" not in page_result:
                        self.logger.warning(f"Page has no words : {idx}")
                        continue
                    for word in page_result["words"]:
                        word_id = word["id"] = str(
                            word["id"]
                        )  # by forcing the type to string, we can avoid the issue with DataRequest parameters(self) -> Dict:
                        word["processor"] = key
                        if word_id not in candidate_words_by_page[idx]:
                            candidate_words_by_page[idx][word_id] = []
                        candidate_words_by_page[idx][word_id].append(word)
                        print(
                            f"candidate_words[idx][word_id] = {len(candidate_words_by_page[idx][word_id])}  > {idx}  {word_id}"
                        )

            print("TEMP OUTPUT 1")
            print(candidate_words_by_page)

            store_json_object(candidate_words_by_page, "/tmp/marie/page_step-1.json")
            # pick the word with the highest confidence
            words_by_confidence_by_page = {}

            for key, candidates in candidate_words_by_page.items():
                idx = key
                self.logger.info(f"Word : {idx}")
                if idx not in words_by_confidence_by_page:
                    words_by_confidence_by_page[idx] = []

                for word_id, words in candidates.items():
                    word = words[0]
                    if len(words) > 1:
                        # pick the word with the highest confidence
                        for word_candidate in words:
                            if word_candidate["confidence"] > word["confidence"]:
                                word = word_candidate

                    words_by_confidence_by_page[idx].append(word)

            print("TEMP OUTPUT 2")
            print(words_by_confidence_by_page)

            store_json_object(
                words_by_confidence_by_page, "/tmp/marie/page_step-2.json"
            )

            min_vote_count = 2
            words_by_vote_by_page = {}

            for key, candidates in candidate_words_by_page.items():
                idx = key
                if idx not in words_by_vote_by_page:
                    words_by_vote_by_page[idx] = []

                for word_id, words in candidates.items():
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

                store_json_object(
                    words_by_vote_by_page, "/tmp/marie/words_by_vote_by_page.json"
                )

            # pick the words by page and update the results
            output_results = deepcopy(default_results)
            for idx, page_result in enumerate(output_results):
                page_result["words"] = words_by_vote_by_page[idx]

            return output_results
