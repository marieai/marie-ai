"""
Using LayoutLMv3 to classify multipage documents
    LayoutLMv3Model - backbone-model to extract features
    LayoutLMv3ForSequenceClassification - LayoutLMv3Model + head (Linear) to classification tasks
Warning: LayoutLMv3 has max of 512 tokens per sample (around 1 A4-page with full of text)
"""
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
from torch import Tensor
from transformers import LayoutLMv3Model
from typing import Any, Optional

from marie.models.multimodal.util.attention_pooling import AttentionPooling

class LayoutLMv3DocumentClassifier(nn.Module):

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 freeze_mode: str = "freeze_all",
                 unfreeze_last_n: Any = None,
                 process_emb_mode: str = "mean",
                 add_page_input: bool = False,
                 num_page_classes: Optional[int] = None,
                 page_emb_dim: int = 64
                 ) -> None:

        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_mode = freeze_mode
        self.unfreeze_last_n = unfreeze_last_n
        self.process_emb_mode = process_emb_mode
        self.add_page_input = add_page_input

        # backbone LayoutLMv3 INIT
        self.backbone = LayoutLMv3Model.from_pretrained(self.model_name)
        self.hidden_size: int = self.backbone.config.hidden_size
        self._set_trainable()

        # optional: page input embedding
        if self.add_page_input:
            assert isinstance(num_page_classes, int), "num_page_classes must be an int (not None)"
            self.page_class_embedding = nn.Embedding(num_page_classes + 1, page_emb_dim, padding_idx=0)
            self.page_class_bridge = nn.Linear(page_emb_dim, self.hidden_size)

        # embedding processor
        self.sequence_encoder = None
        sequence_input_size = self.hidden_size
        if self.add_page_input:
            sequence_input_size += self.hidden_size
        if self.process_emb_mode == "lstm":
            self.sequence_encoder = nn.LSTM(
                input_size=sequence_input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=False,
            )
        elif self.process_emb_mode == "bi_lstm":
            self.sequence_encoder = nn.LSTM(
                input_size=sequence_input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            )
        elif self.process_emb_mode == "attention":
            self.attention_layer = AttentionPooling(sequence_input_size)

        # classifier INIT
        clf_input_size = self.hidden_size * 2 if self.process_emb_mode == "bi_lstm" else self.hidden_size
        if self.add_page_input:
            clf_input_size *= 2
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        **kwargs: Any,
    ) -> "LayoutLMv3DocumentClassifier":
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
        valid_keys = {
            name for name in signature.parameters.keys() if name != "self"
        }
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

    def forward(self, input_ids: Tensor, attention_mask: Tensor, bbox: Tensor, pixel_values: Tensor, page_mask: Tensor,
                page_labels: Optional[Tensor] = None) -> Any:

        if page_mask.sum() == 0:
            raise ValueError("Batch contains documents without valid pages")

        B, P, S = input_ids.shape

        # extract LayoutLMv3 embeddings for batch (all pages in batch)
        # reshape
        input_ids = input_ids.view(B * P, S)
        attention_mask = attention_mask.view(B * P, S)
        bbox = bbox.view(B * P, S, 4)
        pixel_values = pixel_values.view(B * P, 3, pixel_values.size(-2), pixel_values.size(-1))

        # extract embs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox,
                                pixel_values=pixel_values)
        cls_embeddings = outputs.last_hidden_state[:, 0]  # (B * P, hidden_size)
        doc_embeddings = cls_embeddings.view(B, P, -1)  # (B, P, hidden_size) - back to per-document

        # optional: concat page categories embs
        if self.add_page_input and page_labels is not None:
            page_label_emb = self.page_class_embedding(page_labels)  # (B, P, page_emb_dim)
            page_label_emb = self.page_class_bridge(page_label_emb)    # (B, P, hidden_size)
            doc_embeddings = torch.cat([doc_embeddings, page_label_emb], dim=-1)  # (B, P, hidden_size * 2)

        # process embeddings - 1 embedding per each document
        final_embeddings = self._process_emb(page_mask, doc_embeddings)  # (B, hidden)

        # classification layer
        logits = self.classifier(final_embeddings)  # (B, num_classes)
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
            for layer in encoder_layers[-self.unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif self.freeze_mode == "unfreeze_all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {self.freeze_mode}")

    def check_trainable(self) -> None:
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                print("Trainable:", name)

    def _process_emb(self, page_mask: Tensor, doc_embs: Tensor) -> Any:
        """
        LayoutLMv3 embeddings processor based on selected mode. Inclusion of masking of padded pages.
        Modes: 1.mean - mean of embeddings from all pages; 2.lstm - sequence of embeddings.
        """
        if self.process_emb_mode == "lstm":
            assert self.sequence_encoder is not None
            lengths = page_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(doc_embs, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.sequence_encoder(packed)
            final_embedding = h_n.squeeze(0)  # (B, hidden)

        elif self.process_emb_mode == "bi_lstm":
            assert self.sequence_encoder is not None
            lengths = page_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(doc_embs, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.sequence_encoder(packed)
            final_embedding = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2 * hidden)

        elif self.process_emb_mode == "attention":
            final_embedding = self.attention_layer(doc_embs, page_mask)  # (B, hidden)

        elif self.process_emb_mode == "mean":
            page_mask_exp = page_mask.unsqueeze(-1)  # (B, P, 1)
            masked_embed = doc_embs * page_mask_exp
            final_embedding = masked_embed.sum(dim=1) / page_mask_exp.sum(dim=1).clamp(min=1e-9)  # (B, hidden)

        else:
            raise ValueError(f"Unknown embedding processor mode: {self.process_emb_mode}")

        return final_embedding


if __name__ == "__main__":
    model_name_ = 'gordonlim/layoutlmv3-base-finetuned-rvlcdip'
    num_classes_ = 13
    model_obj = LayoutLMv3DocumentClassifier(model_name_,
                                             num_classes_,
                                             freeze_mode='unfreeze_n',
                                             unfreeze_last_n=1,
                                             process_emb_mode="lstm"
                                             )
    model_obj.check_trainable()
