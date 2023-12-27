import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

import marie.models.unilm.layoutreader.s2s_ft.s2s_loader as seq2seq_loader
from marie.models.unilm.layoutreader.s2s_ft.modeling_decoding import (
    BertConfig,
    LayoutlmForSeq2SeqDecoder,
)
from marie.models.unilm.layoutreader.s2s_ft.s2s_loader import Preprocess4Seq2seqDecoder
from marie.models.unilm.layoutreader.s2s_ft.tokenization_minilm import MinilmTokenizer
from marie.models.unilm.layoutreader.s2s_ft.tokenization_unilm import UnilmTokenizer

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
    'layoutlm': BertTokenizer,
}


#  https://arxiv.org/pdf/2108.11591.pdf
#  https://github.com/microsoft/unilm/issues/464


class TextLayout:
    def __init__(self, model_path: str):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # pylint: disable=no-member

        model_type = 'layoutlm'
        max_seq_length = 1024
        do_lower_case = True
        cache_dir = '/mnt/data/marie-ai/model_zoo/unilm/layoutreader/cache'

        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/layoutlmv2-base-uncased",
            do_lower_case=do_lower_case,
            cache_dir=cache_dir if cache_dir else None,
            max_len=max_seq_length,
        )

        config_file = os.path.join(model_path, "config.json")
        config = BertConfig.from_json_file(
            config_file, layoutlm_only_layout_flag=model_type == 'layoutlm'
        )

        max_tgt_length = 511
        pos_shift = False
        self.max_len = max_tgt_length

        self.preprocessor = Preprocess4Seq2seqDecoder(
            list(tokenizer.vocab.keys()),
            tokenizer.convert_tokens_to_ids,
            max_seq_length,
            max_tgt_length=max_tgt_length,
            pos_shift=pos_shift,
            source_type_id=config.source_type_id,
            target_type_id=config.target_type_id,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token,
            layout_flag=model_type == 'layoutlm',
        )

        mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
            [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token]
        )

        forbid_ignore_set = None
        forbid_ignore_word = '.'

        if forbid_ignore_word:
            w_list = []
            for w in forbid_ignore_word.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

        batch_size = 32
        beam_size = 1
        length_penalty = 0
        forbid_duplicate_ngrams = True
        min_len = 1
        ngram_size = 3
        mode = 's2s'
        fp16 = True

        self.beam_size = beam_size

        model = LayoutlmForSeq2SeqDecoder.from_pretrained(
            model_path,
            config=config,
            mask_word_id=mask_word_id,
            search_beam_size=beam_size,
            length_penalty=length_penalty,
            eos_id=eos_word_ids,
            sos_id=sos_word_id,
            forbid_duplicate_ngrams=forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set,
            ngram_size=ngram_size,
            min_len=min_len,
            mode=mode,
            max_position_embeddings=max_seq_length,
            pos_shift=pos_shift,
        )

        n_gpu = torch.cuda.device_count()
        if fp16:
            model.half()

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()

        self.model = model
        self.device = device

    def __call__(self, *args, **kwargs):
        return self.reconstruct(*args, **kwargs)

    def forward(self, words: list[str], boxes: list[[int, int, int, int]]) -> list[int]:
        """
        Re-order words and boxes based on the model prediction

        :param words: Word list [sorted in top-down / left-right fashion for best performance)
        :param boxes: Normalized bounding box list (layoutlm format - x1, y1, x2, y2)
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

            print('traces')
            print(traces)

            if False:
                print(traces.keys())

                pred_seq = traces['pred_seq'][0]
                scores = traces['scores'][0]
                wids = traces['wids'][0]
                ptrs = traces['ptrs'][0]
            #  1,  7,  8, 10,  1,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
            #           4,  4,  4,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4

            if self.beam_size > 1:
                traces = {k: v.tolist() for k, v in traces.items()}
                output_ids = traces['pred_seq'][0]
            else:
                output_ids = traces.squeeze().tolist()
                output_ids = list(np.array(output_ids) - 1)

            # output_ids = traces.squeeze().tolist()
            # output_ids = list(np.array(output_ids) - 1)
            print("output_ids", output_ids)
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
            print("idx", idx)
            if False:
                return words, boxes
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
            raise exception
            logging.warning(exception)
            return words, boxes
