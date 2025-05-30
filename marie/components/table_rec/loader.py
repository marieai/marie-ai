from typing import Optional

import torch

from marie.common.load import ModelLoader
from marie.components.table_rec.model.config import (
    DonutSwinTableRecConfig,
    SuryaTableRecConfig,
    SuryaTableRecDecoderConfig,
)
from marie.components.table_rec.model.encoderdecoder import TableRecEncoderDecoderModel
from marie.components.table_rec.processor import SuryaTableRecProcessor
from marie.settings import settings


class TableRecModelLoader(ModelLoader):
    def __init__(self, checkpoint: Optional[str] = None):
        super().__init__(checkpoint)

        if self.checkpoint is None:
            self.checkpoint = settings.TABLE_REC_MODEL_CHECKPOINT

    def model(
        self, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE
    ) -> TableRecEncoderDecoderModel:
        if device is None:
            device = settings.TORCH_DEVICE_MODEL
        if dtype is None:
            dtype = settings.MODEL_DTYPE

        print('self.checkpoint = ', self.checkpoint)
        config = SuryaTableRecConfig.from_pretrained(self.checkpoint)
        decoder_config = config.decoder
        decoder = SuryaTableRecDecoderConfig(**decoder_config)
        config.decoder = decoder

        encoder_config = config.encoder
        encoder = DonutSwinTableRecConfig(**encoder_config)
        config.encoder = encoder

        model = TableRecEncoderDecoderModel.from_pretrained(
            self.checkpoint, config=config, torch_dtype=dtype
        )

        model = model.to(device)
        model = model.eval()

        if settings.COMPILE_ALL or settings.COMPILE_TABLE_REC:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.config.cache_size_limit = 16
            torch._dynamo.config.suppress_errors = False

            print(
                f"Compiling table recognition model {self.checkpoint} on device {device} with dtype {dtype}"
            )
            compile_args = {'backend': 'openxla'} if device == 'xla' else {}
            model.encoder = torch.compile(model.encoder, **compile_args)
            model.decoder = torch.compile(model.decoder, **compile_args)

        print(
            f"Loaded table recognition model {self.checkpoint} on device {device} with dtype {dtype}"
        )
        return model

    def processor(self) -> SuryaTableRecProcessor:
        processor = SuryaTableRecProcessor(self.checkpoint)

        processor.token_pad_id = 0
        processor.token_eos_id = 1
        processor.token_bos_id = 1
        processor.token_query_end_id = 4
        return processor
