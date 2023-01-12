import os
from typing import Union, List

import numpy as np
import torch
from PIL import Image

from marie.boxes import PSMode
from marie.constants import __model_path__
from marie.logging.logger import MarieLogger
from marie.ocr import CoordinateFormat
from marie.ocr.mock_ocr_engine import MockOcrEngine
from marie.renderer import TextRenderer, PdfRenderer
from marie.renderer.adlib_renderer import AdlibRenderer
from marie.renderer.blob_renderer import BlobRenderer
from marie.utils.image_utils import imwrite
from marie.utils.utils import ensure_exists
from marie.utils.zip_ops import merge_zip


class ExtractPipeline:
    def __init__(
        self,
        models_dir: str = os.path.join(__model_path__),
        cuda: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)
        # self.ocr_engine = DefaultOcrEngine(cuda=use_cuda)
        self.ocr_engine = MockOcrEngine(cuda=use_cuda)
        # self.overlay_processor = OverlayProcessor(
        #     work_dir=ensure_exists("/tmp/form-segmentation"), cuda=use_cuda
        # )

    def __segment(
        self, frames: Union[np.ndarray, List[Image.Image]], root_asset_dir: str
    ):

        clean_dir = ensure_exists(os.path.join(root_asset_dir, "clean"))
        clean_frames = []

        for i, frame in enumerate(frames):
            try:
                doc_id = f"doc_{i}"
                real, mask, clean = self.overlay_processor.segment_frame(doc_id, frame)
                clean_frames.append(clean)

                save_path = os.path.join(clean_dir, f"{doc_id}.tif")
                imwrite(save_path, clean, dpi=(300, 300))
                print(f"Saved clean img : {save_path}")
            except Exception as e:
                self.logger.warning(f"Unable to segment document : {e}")

        return clean_frames

    def execute(
        self,
        frames: Union[np.ndarray, List[Image.Image]],
        pms_mode: PSMode = PSMode.SPARSE,
        coordinate_format: CoordinateFormat = CoordinateFormat.XYWH,
        queue_id: str = None,
        **kwargs,
    ):
        print("Executing pipeline")

        root_asset_dir = "/home/gbugaj/tmp/marie-cleaner/169150505/out"

        # make sure we have clean image
        # clean_frames = self.__segment(frames, root_asset_dir)

        file_id = "gen_"
        clean_frames = frames
        results = self.ocr_engine.extract(clean_frames, pms_mode, coordinate_format)

        self.render_text(clean_frames, results, root_asset_dir)
        self.render_pdf(clean_frames, results, root_asset_dir)
        self.render_blobs(clean_frames, results, root_asset_dir)
        self.render_adlib(clean_frames, results, root_asset_dir)
        self.pack_assets(file_id, root_asset_dir)

    def render_text(self, frames, results, root_asset_dir):
        renderer = TextRenderer(config={"preserve_interword_spaces": True})
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(root_asset_dir, "results.txt"),
        )

    def render_pdf(self, frames, results, root_asset_dir):
        renderer = PdfRenderer(config={})
        renderer.render(
            frames,
            results,
            output_filename=os.path.join(root_asset_dir, "results.pdf"),
        )

    def render_blobs(self, frames, results, root_asset_dir):
        blob_dir = ensure_exists(os.path.join(root_asset_dir, "blobs"))
        print(f"Rendering blob : {blob_dir}")
        renderer = BlobRenderer(config={})
        renderer.render(frames, results, blob_dir)

    def render_adlib(self, frames, results, root_asset_dir):
        adlib_dir = ensure_exists(os.path.join(root_asset_dir, "adlib"))
        print(f"Rendering adlib : {adlib_dir}")
        renderer = AdlibRenderer(config={})
        renderer.render(frames, results, adlib_dir)

    def pack_assets(self, file_id, root_asset_dir):
        # create assets
        blob_dir = ensure_exists(os.path.join(root_asset_dir, "blobs"))
        adlib_dir = ensure_exists(os.path.join(root_asset_dir, "adlib"))

        merge_zip(adlib_dir, os.path.join(root_asset_dir, f"{file_id}.ocr.zip"))
        merge_zip(blob_dir, os.path.join(root_asset_dir, f"{file_id}.blobs.xml.zip"))
