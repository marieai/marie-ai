import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

import marie.models.unilm.layoutreader.s2s_ft.s2s_loader as seq2seq_loader
from marie.models.unilm.layoutreader.s2s_ft.modeling_decoding import (
    BertConfig,
    LayoutlmForSeq2SeqDecoder,
)
from marie.models.unilm.layoutreader.s2s_ft.s2s_loader import Preprocess4Seq2seqDecoder


class TextLayout:
    def __init__(self, model_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # pylint: disable=no-member
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv2-base-uncased"
        )
        config_file = os.path.join(model_path, "config.json")

        self.config = BertConfig.from_json_file(
            config_file, layoutlm_only_layout_flag=False
        )
        self.model = LayoutlmForSeq2SeqDecoder.from_pretrained(
            model_path, config=self.config
        ).to(self.device)
        self.max_len = 511
        self.preprocessor = Preprocess4Seq2seqDecoder(
            list(self.tokenizer.vocab.keys()),
            self.tokenizer.convert_tokens_to_ids,
            1024,
            max_tgt_length=self.max_len,
            layout_flag=True,
        )

    def __call__(self, *args, **kwargs):
        return self.reconstruct(*args, **kwargs)

    def forward(self, words, boxes) -> List[int]:
        """
        :param words: Word list [sorted in top-down / left-right fashion for best performance)
        :param boxes: Normalized bounding box list (layoutlm format)
        :return: Re-ordered index list
        """
        assert len(words) == len(boxes)

        instance = [[x[0], *x[1]] for x in list(zip(words, boxes))], len(boxes)
        instances = [self.preprocessor(instance)]
        with torch.no_grad():
            batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
            batch = [t.to(self.device) if t is not None else None for t in batch]
            (
                input_ids,
                token_type_ids,
                position_ids,
                input_mask,
                mask_qkv,
                task_idx,
            ) = batch

            traces = self.model(
                input_ids,
                token_type_ids,
                position_ids,
                input_mask,
                task_idx=task_idx,
                mask_qkv=mask_qkv,
            )
            output_ids = traces.squeeze().tolist()
            output_ids = list(np.array(output_ids) - 1)
            return output_ids

    def reconstruct(
        self, words: List[str], boxes: List[List[int]]
    ) -> Tuple[List[str], List[List[int]]]:

        assert len(words) == len(boxes)

        if len(words) > self.max_len:
            logging.warning(
                f"Page contains {len(words)} words. Exceeds the {self.max_len} limit and will not be reordered."
            )
            return words, boxes

        try:
            idx = self.forward(words, boxes)
            processed_idx = list(dict.fromkeys(idx))
            if len(processed_idx) != len(words):
                processed_idx = [idx for idx in processed_idx if idx < len(words)]
                unused_idx = sorted(
                    list(set(range(len(words))) - set(processed_idx[: len(words)]))
                )
                logging.info(
                    f"There is {len(words)} words but only {len(processed_idx)} indexes. "
                    f"Unmatched indexes: {unused_idx}"
                )
                processed_idx.extend(unused_idx)
                logging.info(
                    f"There is now {len(words)} wordsand {len(processed_idx)} indexes."
                )
                assert len(processed_idx) == len(words)

            words = list(np.array(words)[processed_idx])
            boxes = [elem.tolist() for elem in np.array(boxes)[processed_idx]]
            return words, boxes

        except Exception as exception:  # pylint: disable=broad-except
            logging.warning(exception)
            return words, boxes
