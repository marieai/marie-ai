import os
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from PIL import Image
from torch import nn

from marie.logging.logger import MarieLogger
from marie.models.utils import initialize_device_settings

from ...embeddings.jina.jina_embeddings import JinaEmbeddings
from ...embeddings.openai.openai_embeddings import OpenAIEmbeddings
from ...utils.overlap import merge_bboxes_as_block
from ...utils.resize_image import resize_image
from ...utils.utils import ensure_exists
from .base import BaseTemplateMatcher
from .model import TemplateMatchResult


class MetaTemplateMatcher(BaseTemplateMatcher):
    """
    This class is used to match a template in an image using the Pattern Matching algorithm.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        labels: Optional[List[str]] = None,
        batch_size: int = 16,
        use_auth_token: Optional[Union[str, bool]] = None,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
        show_error: Optional[Union[str, bool]] = True,
        **kwargs,
    ):
        """

        :param model_name_or_path: Directory of a saved model or the name of a public model  from the HuggingFace model hub.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param use_gpu: Whether to use GPU (if available).
        :param batch_size: Number of Documents to be processed at a time.
        :param use_auth_token: The API token used to download private models from Huggingface.
                               If this parameter is set to `True`, then the token generated when running
                               `transformers-cli login` (stored in ~/.huggingface) will be used.
                               Additional information can be found here
                               https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
        :param devices: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
                        A list containing torch device objects and/or strings is supported (For example
                        [torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
                        parameter is not used and a single cpu device is used for inference.
        """
        super().__init__(False, **kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Document matcher : {model_name_or_path}")
        self.show_error = show_error  # show prediction errors
        self.batch_size = batch_size
        self.labels = labels
        self.progress_bar = False

        resolved_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if len(resolved_devices) > 1:
            self.logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                resolved_devices[0],
            )

        self.device = resolved_devices[0]
        self.embedding_cache = {}
        self.embeddings_processor = JinaEmbeddings(
            model_name_or_path="hf://jinaai/jina-embeddings-v2-base-en"
        )
        self.embeddings_processor_clip = OpenAIEmbeddings(
            model_name_or_path="marie/clip-snippet-rn50x4"
            # model_name_or_path="hf://openai/clip-vit-base-patch32"
        )
        self.cached_embeddings_clips = {}

        self.enable_visualization = True

    def predict(
        self,
        frame: np.ndarray,
        template_frames: list[np.ndarray],
        template_boxes: list[tuple[int, int, int, int]],
        template_labels: list[str],
        template_texts: list[str] = None,
        score_threshold: float = 0.9,
        scoring_strategy: str = "weighted",
        max_objects: int = 1,
        batch_size: int = 1,
        words: list[str] = None,
        word_boxes: list[tuple[int, int, int, int]] = None,
        word_lines: list[tuple[int, int, int, int]] = None,
    ) -> list[TemplateMatchResult]:

        predictions = []

        if words is None or word_boxes is None:
            self.logger.warning("No words or word_boxes provided. Skipping prediction.")
            return predictions

        assert len(words) == len(word_boxes)
        assert len(template_frames) == len(template_boxes)
        assert len(template_texts) == len(template_labels)
        # extract type (EXACT, REGEX, FUZZY, EMBEDDINGS) from the template_labels

        page_words = words
        page_boxes = word_boxes
        word_lines = word_lines
        k = 0
        min_word_length = 3

        for idx, (template_text, template_label) in enumerate(
            zip(template_texts, template_labels)
        ):
            if template_text is None or len(template_text) < min_word_length:
                self.logger.debug(
                    f"Skipping template_text {template_text} as it is too short : {min_word_length}"
                )
                continue

            template_bbox = template_boxes[idx]
            temp_x, temp_y, temp_w, temp_h = template_bbox

            temp_x = int(max(temp_x, 0))
            temp_y = int(max(temp_y, 0))
            template_image = template_frames[idx]
            fragment = template_image[
                temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
            ]

            cv2.imwrite(f"/tmp/dim/meta_{idx}_fragment.png", fragment)

            ngram = len(template_text.split(" "))
            ngrams = [ngram - 1, ngram, ngram + 1]
            ngrams = [n for n in ngrams if 0 < n <= len(page_words)]

            candidates = []
            for ngram in ngrams:
                for i in range(len(page_words) - ngram + 1):
                    ngram_words = page_words[i : i + ngram]
                    ngram_boxes = page_boxes[i : i + ngram]

                    # # each ngram word should be at least 3 characters
                    # if any(len(w) < min_word_length for w in ngram_words):
                    #     self.logger.debug(
                    #         f"Skipping ngram {ngram_words} as it is too short : {min_word_length}"
                    #     )
                    #     continue
                    # TODO add check for distance between words in ngram

                    if word_lines:
                        ngram_lines = word_lines[i : i + ngram]
                        if len(set(ngram_lines)) > 1:
                            self.logger.debug(
                                f"Skipping ngram {ngram_words} as it is not in the same line"
                            )
                            continue

                    key = "_".join(ngram_words)
                    box = merge_bboxes_as_block(ngram_boxes)
                    x, y, w, h = box
                    ngram_snippet = frame[y : y + h, x : x + w :]

                    ngram_words = " ".join(ngram_words).strip().upper()
                    template_text = template_text.strip().upper()
                    sim_val = round(
                        self.score(ngram_words, template_text, ngram_snippet, fragment),
                        3,
                    )

                    if ngram_words == template_text or sim_val > score_threshold:
                        candidates.append(
                            {
                                "ngram": ngram,
                                "words": ngram_words,
                                "similarity": sim_val,
                                "candidate": TemplateMatchResult(
                                    bbox=box,
                                    label=template_label,
                                    score=sim_val,
                                    similarity=sim_val,
                                ),
                            }
                        )

                        if self.enable_visualization:
                            ensure_exists(f"/tmp/fragments/meta/{key}")
                            cv2.imwrite(
                                f"/tmp/fragments/meta/{k}_{round(sim_val, 3)}.png",
                                ngram_snippet,
                            )
                        k += 1

            # choose the most specific prediction group
            if candidates:
                sorted_candidates = sorted(
                    candidates,
                    key=lambda x: (x['ngram'], x['similarity']),
                    reverse=False,
                )
                most_specific_prediction = sorted_candidates[0]
                candidates = [
                    x
                    for x in sorted_candidates
                    if x['ngram'] == most_specific_prediction['ngram']
                ]
                for prediction in candidates:
                    predictions.append(prediction["candidate"])

        return predictions

    def get_embedding(self, text):
        if text not in self.embedding_cache:
            max_length = 1024
            if len(text) > max_length:
                text = text[:max_length]
            self.embedding_cache[text] = self.embeddings_processor.get_embeddings(
                [text], truncation=True, max_length=max_length
            ).embeddings[0]
        return self.embedding_cache[text]

    def get_embedding_image(
        self, image: np.ndarray, words: list, boxes: list
    ) -> np.ndarray:
        key = image.tobytes()
        if key in self.cached_embeddings_clips:
            return self.cached_embeddings_clips[key]

        # This is a pre-processing step to get the embeddings for the words and boxes in the image
        # this is critical for the similarity calculation that we resize with PADDING and not just scaling
        # image = resize_image(image, (224, 224))[0]
        if image.shape[0] != 224 or image.shape[1] != 224:
            raise ValueError("Image must be 224x224")

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        embedding = self.embeddings_processor_clip.get_embeddings(
            texts=words, boxes=boxes, image=image
        )

        self.cached_embeddings_clips[key] = embedding.embeddings
        return embedding.embeddings

    def score(
        self,
        ngram_words: str,
        template_text: str,
        query_snippet: np.ndarray,
        template_snippet: np.ndarray,
    ) -> float:
        from Levenshtein import distance
        from numpy.linalg import norm

        d = distance(ngram_words, template_text)
        sim_val = 1 - d / max(len(ngram_words), len(template_text))
        if sim_val < 0.5:
            return sim_val

        t_clip = resize_image(template_snippet, (224, 224))[0]
        q_clip = resize_image(query_snippet, (224, 224))[0]

        template_snippet_features = self.get_embedding_image(t_clip, words=[], boxes=[])
        query_pred_snippet_features = self.get_embedding_image(
            q_clip, words=[], boxes=[]
        )

        query_embedding = self.get_embedding(ngram_words)
        template_embedding = self.get_embedding(template_text)

        cosine = nn.CosineSimilarity(dim=1)
        embedding_sim = cosine(
            torch.from_numpy(template_snippet_features).view(1, -1),
            torch.from_numpy(query_pred_snippet_features).view(1, -1),
        )
        embedding_sim = embedding_sim.cpu().numpy()[0]

        cos_sim_val = cosine(
            torch.from_numpy(query_embedding).view(1, -1),
            torch.from_numpy(template_embedding).view(1, -1),
        )
        cos_sim_val = cos_sim_val.cpu().numpy()[0]
        total_sim = (sim_val + cos_sim_val + embedding_sim) / 3
        sout = f"similarity : {sim_val:<10} - {cos_sim_val:<10} > {embedding_sim:<10} ---- {total_sim:<10} --- {ngram_words}"
        self.logger.info(sout)
        print(sout)
        return total_sim
