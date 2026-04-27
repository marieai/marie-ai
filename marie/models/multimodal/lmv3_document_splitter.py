from typing import Any

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import transformers

from marie.models.multimodal.util.attention_pooling import AttentionPooling


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
        self.backbone = transformers.LayoutLMv3Model.from_pretrained(self.model_name)
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

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> "LayoutLMv3DocumentSplitter":
        model_path = Path(model_name_or_path)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_kwargs = dict(config.get("custom_model_parameters", {}))
        model_kwargs.update(kwargs)
        model_kwargs = cls._filter_constructor_kwargs(model_kwargs)
        cls._ensure_required_kwargs(model_kwargs)

        model = cls(**model_kwargs)
        checkpoint_path = cls._resolve_checkpoint_path(model_path)
        state_dict = cls._load_state_dict(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.config = SimpleNamespace(id2label=config.get("id2label", {}))
        return model

    @classmethod
    def _filter_constructor_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(cls.__init__)
        valid_keys = {name for name in signature.parameters.keys() if name != "self"}
        return {k: v for k, v in kwargs.items() if k in valid_keys}

    @classmethod
    def _ensure_required_kwargs(cls, kwargs: dict[str, Any]) -> None:
        signature = inspect.signature(cls.__init__)
        missing = []
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.default is inspect.Parameter.empty and name not in kwargs:
                missing.append(name)
        if missing:
            raise ValueError(
                f"Missing required custom model parameters for {cls.__name__}: "
                f"{', '.join(missing)}"
            )

    @staticmethod
    def _resolve_checkpoint_path(model_path: Path) -> Path:
        for candidate in ("model.safetensors", "pytorch_model.bin"):
            checkpoint = model_path / candidate
            if checkpoint.exists():
                return checkpoint
        raise FileNotFoundError(
            f"Could not find model weights in {model_path}. "
            "Expected one of: model.safetensors, pytorch_model.bin"
        )

    @staticmethod
    def _load_state_dict(
        checkpoint_path: Path, map_location: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        if checkpoint_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Loading .safetensors checkpoints requires the `safetensors` package."
                ) from exc
            return load_file(str(checkpoint_path), device=str(map_location))

        raw_checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        if isinstance(raw_checkpoint, dict):
            return (
                raw_checkpoint.get("state_dict")
                or raw_checkpoint.get("model_state_dict")
                or raw_checkpoint
            )
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}: "
            f"{type(raw_checkpoint)}"
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
