from typing import Any

import torch
import torch.nn as nn
from transformers import LayoutLMv3Model

from marie.components.document_classifier.models.util.attention_pooling import (
    AttentionPooling,
)


class LayoutLMv3DocumentSplitter(nn.Module):

    def __init__(
        self,
        model_name: str,
        freeze_mode: str = "freeze_all",
        unfreeze_last_n: int | None = None,
        process_emb_mode: str = "bi_lstm",
        context_pages_num: int = 5,
    ) -> None:

        super().__init__()
        self.model_name = model_name
        self.num_classes = 2
        self.freeze_mode = freeze_mode
        self.unfreeze_last_n = unfreeze_last_n
        self.process_emb_mode = process_emb_mode
        self.context_len = context_pages_num

        # backbone LayoutLMv3 INIT
        self.backbone = LayoutLMv3Model.from_pretrained(self.model_name)
        self.hidden_size: int = self.backbone.config.hidden_size
        self._set_trainable()
        self.sequence_encoder = None

        if self.process_emb_mode == "lstm":
            self.sequence_encoder = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=False,
            )
        elif self.process_emb_mode == "bi_lstm":
            self.sequence_encoder = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            )
        elif self.process_emb_mode == "attention":
            self.attention_layer = AttentionPooling(self.hidden_size)

        # classifier INIT (boundary: yes/no)
        clf_input_size = (
            self.hidden_size * 4
            if self.process_emb_mode == "bi_lstm"
            else self.hidden_size
        )
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        bbox: Any,
        pixel_values: Any,
        page_mask: Any,
        center_page_idx: Any,
    ) -> Any:
        """
        input_ids, attention_mask, bbox, pixel_values: (B, P, S, ...)
        page_mask: (B, P)
        """
        B, P, S = input_ids.shape

        # extract LayoutLMv3 embeddings for batch (all pages in batch)
        # reshape
        input_ids = input_ids.view(B * P, S)
        attention_mask = attention_mask.view(B * P, S)
        bbox = bbox.view(B * P, S, 4)
        pixel_values = pixel_values.view(
            B * P, 3, pixel_values.size(-2), pixel_values.size(-1)
        )

        # extract emb
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        cls_embeddings = outputs.last_hidden_state[:, 0]  # (B * P, hidden_size)
        doc_embeddings = cls_embeddings.view(B, P, -1)  # (B, P, hidden_size)

        # process embeddings
        final_embeddings = self._process_emb(
            page_mask, doc_embeddings, center_page_idx
        )  # (B, hid_size) or (B, 2*hid)

        # classification layer
        logits = self.classifier(final_embeddings)  # (B, 2)
        return logits

    def _set_trainable(self) -> None:
        """Manage pre-trained backbone freezing"""

        # init freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        # find encoder layers
        if hasattr(self.backbone, "encoder"):
            encoder_layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layer"):
            encoder_layers = self.backbone.layer
        else:
            raise ValueError("Unknown encoder structure")

        # unfreeze right part
        if self.freeze_mode == "freeze_all":
            pass
        elif self.freeze_mode == "unfreeze_n":
            assert self.unfreeze_last_n is not None
            for layer in encoder_layers[-self.unfreeze_last_n :]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif self.freeze_mode == "unfreeze_all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {self.freeze_mode}")

    def _process_emb(self, page_mask: Any, doc_embs: Any, center_page_idx: Any) -> Any:
        """LayoutLMv3 embeddings processor based on selected mode. Inclusion of masking of padded pages.
        Modes: 1.mean - mean of embeddings from all pages; 2.lstm - sequence of embeddings.
        """
        assert doc_embs.size(1) == self.context_len, (
            f"Expected exactly {self.context_len} pages in the sequence. "
            f"Got:{doc_embs.size(1)}"
        )

        if self.process_emb_mode == "lstm":
            # lstm - sequence of embeddings for each page sequence in batch
            lengths = page_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                doc_embs, lengths, batch_first=True, enforce_sorted=False
            )
            assert self.sequence_encoder is not None
            _, (h_n, _) = self.sequence_encoder(packed)
            final_embedding = h_n.squeeze(0)  # (B, hidden)

        elif self.process_emb_mode == "bi_lstm":
            lengths = page_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                doc_embs, lengths, batch_first=True, enforce_sorted=False
            )
            assert self.sequence_encoder is not None
            output, _ = self.sequence_encoder(packed)
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )  # (B, P, 2*hidden)
            b = unpacked.size(0)
            batch_idx = torch.arange(b, device=unpacked.device)
            curr_repr = unpacked[batch_idx, center_page_idx, :]  # (B, 2*hidden)
            next1_repr = unpacked[batch_idx, center_page_idx + 1, :]  # (B, 2*hidden)
            final_embedding = torch.cat(
                [curr_repr, next1_repr], dim=1
            )  # (B, 4 * hidden)

        elif self.process_emb_mode == "attention":
            final_embedding = self.attention_layer(doc_embs, page_mask)

        elif self.process_emb_mode == "mean":
            page_mask_exp = page_mask.unsqueeze(-1)  # (B, P, 1)
            masked_emb = doc_embs * page_mask_exp
            final_embedding = masked_emb.sum(dim=1) / page_mask_exp.sum(dim=1).clamp(
                min=1e-9
            )

        else:
            raise ValueError(
                f"Unknown embedding processor mode: {self.process_emb_mode}"
            )

        return final_embedding
