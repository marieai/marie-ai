import argparse
from typing import List, Optional

import PIL.Image
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
import torch
from cvat_sdk import make_client
from cvat_sdk.auto_annotation import DetectionFunctionSpec, DetectionFunction
from cvat_sdk.auto_annotation.driver import (
    _AnnotationMapper,
    _DetectionFunctionContextImpl,
)
from cvat_sdk.core import Client
from cvat_sdk.core.progress import NullProgressReporter, ProgressReporter
from cvat_sdk.datasets import TaskDataset

from marie.boxes import BoxProcessorUlimDit, PSMode


class TorchvisionDetectionFunction:
    def __init__(self, **kwargs) -> None:
        # load the ML model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latest_image = None
        self.predictor = BoxProcessorUlimDit(
            # models_dir="/etc/marie/model_zoo/unilm/dit/text_detection",
            models_dir="/mnt/data/marie-ai/model_zoo/unilm/dit/text_detection",
            cuda=torch.cuda.is_available(),
        )

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        # describe the annotations
        return cvataa.DetectionFunctionSpec(labels=[cvataa.label_spec("bbox", 0)])

    def detect(
        self, context, image: PIL.Image.Image
    ) -> List[models.LabeledShapeRequest]:
        print("image", type(image), image.size)

        # convert the input into a form the model can understand
        (
            boxes,
            fragments,
            lines,
            _,
            lines_bboxes,
        ) = self.predictor.extract_bounding_boxes(
            "cvat",
            "field",
            image,
            PSMode.SPARSE,
            bbox_optimization=False,
            bbox_context_aware=False,
        )

        box_format = "xywh"
        results = []
        label_id = 0

        for box in boxes:
            # box = [int(v) for v in box]
            if box_format == "xywh":
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                # convert box to cvat float format
                box = [float(v) for v in box]
                spec = cvataa.rectangle(label_id, box)
                results.append(spec)

        return results


def annotate_task_frame(
    client: Client,
    task_id: int,
    frame_id: int,
    function: DetectionFunction,
    *,
    pbar: Optional[ProgressReporter] = None,
    clear_existing: bool = False,
    allow_unmatched_labels: bool = False,
) -> None:
    """
    Downloads data for the task with the given ID, applies the given function to it
    and uploads the resulting annotations back to the task.

    Only tasks with 2D image (not video) data are supported at the moment.

    client is used to make all requests to the CVAT server.

    Currently, the only type of auto-annotation function supported is the detection function.
    A function of this type is applied independently to each image in the task.
    The resulting annotations are then combined and modified as follows:

    * The label IDs are replaced with the IDs of the corresponding labels in the task.
    * The frame numbers are replaced with the frame number of the image.
    * The sources are set to "auto".

    See the documentation for DetectionFunction for more details.

    If the function is found to violate any constraints set in its interface, BadFunctionError
    is raised.

    pbar, if supplied, is used to report progress information.

    If clear_existing is true, any annotations already existing in the tesk are removed.
    Otherwise, they are kept, and the new annotations are added to them.

    The allow_unmatched_labels parameter controls the behavior in the case when a detection
    function declares a label in its spec that has no corresponding label in the task.
    If it's set to true, then such labels are allowed, and any annotations returned by the
    function that refer to this label are ignored. Otherwise, BadFunctionError is raised.
    """

    if pbar is None:
        pbar = NullProgressReporter()

    dataset = TaskDataset(client, task_id)

    assert isinstance(function.spec, DetectionFunctionSpec)

    mapper = _AnnotationMapper(
        client.logger,
        function.spec.labels,
        dataset.labels,
        allow_unmatched_labels=allow_unmatched_labels,
    )

    shapes = []

    with pbar.task(total=len(dataset.samples), unit="samples"):
        for sample in pbar.iter(dataset.samples):
            if sample.frame_index != frame_id:
                continue
            client.logger.info("Processing frame %d", sample.frame_index)
            frame_shapes = function.detect(
                _DetectionFunctionContextImpl(sample.frame_name),
                sample.media.load_image(),
            )
            mapper.validate_and_remap(frame_shapes, sample.frame_index)
            shapes.extend(frame_shapes)

    client.logger.info("Uploading annotations to task %d", task_id)
    clear_existing = False

    if clear_existing:
        client.tasks.api.update_annotations(
            task_id,
            task_annotations_update_request=models.LabeledDataRequest(shapes=shapes),
        )
    else:
        client.tasks.api.partial_update_annotations(
            "create",
            task_id,
            patched_labeled_data_request=models.PatchedLabeledDataRequest(
                shapes=shapes
            ),
        )


def main(task_id: int, frame_id: int, host: str, port: int, user: str, password: str):
    with make_client(host=host, port=port, credentials=(user, password)) as client:
        annotate_task_frame(
            client,
            task_id,
            frame_id,
            TorchvisionDetectionFunction(box_score_thresh=0.5),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate a frame of a task using a detection function"
    )
    parser.add_argument("--task", type=int, help="ID of the task to annotate")
    parser.add_argument("--frame", type=int, help="ID of the frame to annotate")
    # CVAT
    parser.add_argument("--host", type=str, help="CVAT host")
    parser.add_argument("--port", type=str, help="CVAT port")
    parser.add_argument("--user", type=str, help="CVAT user")
    parser.add_argument("--password", type=str, help="CVAT password")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args.task, args.frame, args.host, args.port, args.user, args.password)
