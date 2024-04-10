import base64
import io
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from docarray import DocList
from docarray.base_doc.doc import BaseDocWithoutId
from PIL import Image

from marie import Executor, requests, safely_encoded
from marie.api.docs import BaseDoc
from marie.components.template_matching import BaseTemplateMatcher, VQNNFTemplateMatcher
from marie.components.template_matching.model import TemplateMatchResult
from marie.logging.logger import MarieLogger
from marie.logging.predefined import default_logger as logger
from marie.models.utils import setup_torch_optimizations
from marie.utils.docs import docs_from_asset, docs_from_file, frames_from_docs
from marie.utils.network import get_ip_address
from marie.utils.resize_image import resize_image


class TemplateMatchResultDoc(BaseDocWithoutId, frozen=True):
    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    similarity: float
    frame_index: Optional[int] = 0


class TemplateMatchingResultDoc(BaseDoc, frozen=True):
    asset_key: str
    results: List[TemplateMatchResultDoc]


class TemplateSelector(BaseDocWithoutId, frozen=True):
    region: List[int]
    frame: str
    bbox: List[int]
    label: str
    text: str
    create_window: bool
    top_k: int


class TemplateMatchingRequestDoc(BaseDoc):
    asset_key: str
    id: str
    pages: List[int]
    score_threshold: float
    scoring_strategy: str
    max_overlap: float
    window_size: List[int]
    matcher: str
    downscale_factor: float
    selectors: List[TemplateSelector]


def convert_to_protobuf_doc(match: TemplateMatchResult) -> TemplateMatchResultDoc:
    """
    Convert a TemplateMatchResult to a TemplateMatchResultDoc
    :param match:
    :return: protobuf serializable TemplateMatchResultDoc
    """
    return TemplateMatchResultDoc(
        bbox=match.bbox,
        label=match.label,
        score=match.score,
        similarity=match.similarity,
        frame_index=match.frame_index,
    )


class TemplateMatchingExecutor(Executor):
    def __init__(
        self,
        name: str = "",
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        pipeline: Optional[dict[str, any]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        """
        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param dtype: inference data type, if None defaults to torch.float32 if device == 'cpu' else torch.float16.
        """
        super().__init__(**kwargs)
        import time

        logger.info(f"Starting mock executor : {time.time()}")
        setup_torch_optimizations()

        self.show_error = True  # show prediction errors
        # sometimes we have CUDA/GPU support but want to only use CPU
        use_cuda = torch.cuda.is_available()
        if os.environ.get("MARIE_DISABLE_CUDA"):
            use_cuda = False
        self.logger = MarieLogger(context=self.__class__.__name__)

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not use_cuda:
            device = "cpu"
        self.device = device

        self.runtime_info = {
            "name": self.__class__.__name__,
            "instance_name": kwargs.get("runtime_args", {}).get("name", "not_defined"),
            "model": "",
            "host": get_ip_address(),
            "workspace": self.workspace,
            "use_cuda": use_cuda,
            "device": self.device.__str__() if self.device is not None else "",
        }

        logger.info(f"Runtime info: {self.runtime_info}")
        logger.info(f"Pipeline : {pipeline}")

    def convert_template_selectors(
        self,
        selectors: List[TemplateSelector],
        window_size: Union[List[int], tuple[int, int]],
    ):
        """
        Convert TemplateSelector to Template Matching Selectors
        :param selectors:
        :param window_size:
        :return:
        """

        print(f"Converting {len(selectors)} selector(s)")
        template_frames = []
        template_bboxes = []
        template_labels = []
        template_texts = []

        for i, selector in enumerate(selectors):
            buf = io.BytesIO(base64.b64decode(selector.frame))
            image = Image.open(buf)
            image = image.convert("RGB")
            frame = np.array(image)
            frame = frame[:, :, ::-1].copy()

            boxes_xywh = [
                selector.bbox
            ]  # currently only one bbox is supported per selector
            region = selector.region
            label = selector.label
            text = selector.text

            if selector.create_window:
                frame, coord = resize_image(
                    frame,
                    window_size,
                    keep_max_size=True,
                )
                boxes_xywh = [coord]

            (
                sel_template_frames,
                sel_template_bboxes,
            ) = BaseTemplateMatcher.extract_windows(
                frame, boxes_xywh, window_size, allow_padding=True
            )

            template_frames.extend(sel_template_frames)
            template_bboxes.extend(sel_template_bboxes)
            template_labels.append(label)
            template_texts.append(text)
            assert (
                len(template_frames)
                == len(template_bboxes)
                == len(template_labels)
                == len(template_texts)
            )

            for template_frame in template_frames:
                cv2.imwrite(
                    f"/tmp/dim/template/template_frame_SELECTOR_{i}.png", template_frame
                )

        return template_frames, template_bboxes, template_labels, template_texts

    @requests(on="/document/matcher")
    def match(
        self,
        docs: DocList[TemplateMatchingRequestDoc],
        parameters: dict,
        *args,
        **kwargs,
    ):
        print("TEMPLATE MATCHING EXECUTOR")
        if docs is None or len(docs) == 0:
            return {"error": "empty payload"}

        print("Dumping docs:")
        if False:
            for doc in docs:
                print(doc)

        if len(docs) > 1:
            return {"error": "expected single document"}
        pages = docs[0].pages
        if docs[0].pages is None or len(docs[0].pages) == 0 or docs[0].pages[0] == -1:
            pages = None
        has_pages = pages is not None and len(pages) > 0
        dff = docs_from_file(docs[0].asset_key, pages=pages)
        frames_from_file = frames_from_docs(dff)

        assert len(frames_from_file) > 0

        matcher = VQNNFTemplateMatcher(model_name_or_path="NONE")
        doc = docs[0]

        logger.info("Matching parameters ************")
        logger.info(f"Asset Key: {doc.asset_key}")
        logger.info(f"Selectors : {len(doc.selectors)}")
        logger.info(f"Mode: {doc.matcher}")
        logger.info(f"Scoring strategy: {doc.scoring_strategy}")
        logger.info(f"Score threshold: {doc.score_threshold}")
        logger.info(f"Window size: {doc.window_size}")
        logger.info(f"Max overlap: {doc.max_overlap}")
        logger.info(f"Downscale factor (rescale) : {doc.downscale_factor}")
        logger.info(f"Pages: {pages}")
        logger.info(f"Has pages: {has_pages}")

        if len(doc.selectors) == 0:
            return {"error": "selectors not present"}

        (
            template_frames,
            template_bboxes,
            template_labels,
            template_texts,
        ) = self.convert_template_selectors(doc.selectors, doc.window_size)

        results = matcher.run(
            frames=frames_from_file,
            # TODO: convert to Pydantic model
            template_frames=template_frames,
            template_boxes=template_bboxes,
            template_labels=template_labels,
            template_texts=template_texts,
            metadata=None,
            score_threshold=doc.score_threshold,
            scoring_strategy=doc.scoring_strategy,
            max_overlap=0.5,
            max_objects=2,
            window_size=(doc.window_size[0], doc.window_size[1]),
            downscale_factor=doc.downscale_factor,
        )

        print("Results:")
        print(results)

        for result in results:
            result.frame_index = (
                pages[result.frame_index] if has_pages else result.frame_index
            )

        print("Results after page conversion:")
        print(results)

        # we only return one doc with the results
        reply = DocList[TemplateMatchingResultDoc]()
        reply.append(
            TemplateMatchingResultDoc(
                asset_key=doc.asset_key,
                results=[convert_to_protobuf_doc(result) for result in results],
            )
        )

        return reply

        tmr = TemplateMatchResult(
            bbox=(10, 20, 40, 100),
            label="LABELA ABC",
            score=0.9,
            similarity=0.6,
            frame_index=0,
        )

        reply = DocList[TemplateMatchingResultDoc]()

        reply.append(
            TemplateMatchingResultDoc(
                asset_key="RETURN_ASSET_KEY",
                results=[
                    convert_to_protobuf_doc(tmr),
                    convert_to_protobuf_doc(tmr),
                ],
            )
        )

        return reply

        if len(docs) == 0:
            return {"error": "empty payload"}
        if len(docs) > 1:
            return {"error": "expected single document"}

        doc = docs[0]
        # load documents from specified document asset key
        docs = docs_from_asset(doc.asset_key, doc.pages)

        for doc in docs:
            print(doc.id)

        frames = frames_from_docs(docs)
        frame_len = len(frames)

        print(f"{frame_len=}")

        import time

        if "payload" not in parameters or parameters["payload"] is None:
            return {"error": "empty payload"}
        else:
            payload = parameters["payload"]
        regions = payload["regions"] if "regions" in payload else []
        for region in regions:
            region["id"] = int(region["id"])
            region["pageIndex"] = int(region["pageIndex"])

        np_arr = np.array([1, 2, 3])
        out = [
            {"sample": 112, "complex": ["a", "b"]},
            {"sample": 112, "complex": ["a", "b"], "np_arr": np_arr},
        ]

        time.sleep(1)
        # invoke the safely_encoded decorator as a function
        meta = get_ip_address()
        #  DocList / Dict / `None`
        converted = safely_encoded(lambda x: x)(self.runtime_info)
        return converted
