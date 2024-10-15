from abc import ABC
from typing import List

from docarray import DocList

from marie.excepts import BadConfigSource
from marie.logging_core.logger import MarieLogger
from marie.logging_core.profile import TimeContext
from marie.ocr.util import get_words_and_boxes
from marie.pipe import (
    ClassifierPipelineComponent,
    NamedEntityPipelineComponent,
    PipelineComponent,
    PipelineContext,
)
from marie.pipe.components import load_pipeline
from marie.pipe.voting import ClassificationResult, get_voting_strategy
from marie.utils.docs import docs_from_image


class BasePipeline(ABC):
    def __init__(
        self,
        silence_exceptions: bool = False,
        **kwargs,
    ) -> None:
        self.show_error = True  # show prediction errors
        self.silence_exceptions = silence_exceptions
        self.logger = MarieLogger(context=self.__class__.__name__)

    def reload_pipeline(
        self, pipeline_name, pipelines_config
    ) -> tuple[str, dict, dict]:
        with TimeContext(f"### Reloading pipeline : {pipeline_name}", self.logger):
            try:
                self.logger.info(f"Reloading pipeline : {pipeline_name}")
                if pipelines_config is None:
                    raise BadConfigSource(
                        "Invalid pipeline configuration, no pipelines found"
                    )

                pipeline_config = None
                for conf in pipelines_config:
                    conf = conf["pipeline"]
                    if conf.get("name") == pipeline_name:
                        pipeline_config = conf
                        break

                if pipeline_config is None:
                    raise BadConfigSource(
                        f"Invalid pipeline configuration, pipeline not found : {pipeline_name}"
                    )

                (
                    pipeline_name,
                    classifier_groups,
                    indexer_groups,
                ) = load_pipeline(pipeline_config)
                self.logger.info(f"Reloaded successfully pipeline : {pipeline_name} ")

                return pipeline_name, classifier_groups, indexer_groups
            except Exception as e:
                self.logger.error(f"Error reloading pipeline : {e}")
                raise e

    def execute_classifier_and_indexer_pipeline(
        self,
        frames,
        metadata,
        ocr_results,
        pipeline_name,
        classifier_groups,
        indexer_groups,
        page_indexer_enabled,
    ):

        if "classifications" not in metadata:
            metadata["classifications"] = []
        if "indexers" not in metadata:
            metadata["indexers"] = []

        for group, classifier_group in classifier_groups.items():
            self.logger.info(
                f"Processing classifier pipeline/group :  {pipeline_name}, {group}"
            )
            document_classifiers = classifier_group["classifiers"]
            sub_classifiers = (
                classifier_group["sub_classifiers"]
                if "sub_classifiers" in classifier_group
                else {}
            )

            processing_pipeline = [
                ClassifierPipelineComponent(
                    name="classifier_pipeline",
                    document_classifiers=document_classifiers,
                )
            ]

            if page_indexer_enabled:
                if group in indexer_groups:
                    document_indexers = indexer_groups[group]["indexer"]
                    processing_pipeline.append(
                        NamedEntityPipelineComponent(
                            name="ner_pipeline_component",
                            document_indexers=document_indexers,
                        )
                    )

            results = self.execute_pipeline(
                processing_pipeline, sub_classifiers, frames, ocr_results
            )

            metadata["classifications"].append(
                {
                    "group": group,
                    "classification": (
                        results["classifier"] if "classifier" in results else {}
                    ),
                }
            )

            metadata["indexers"].append(
                {
                    "group": group,
                    "indexer": results["indexer"] if "indexer" in results else {},
                }
            )

    def execute_pipeline(
        self,
        processing_pipeline: List[PipelineComponent],
        sub_classifiers: dict[str, any],
        frames: List,
        ocr_results: dict,
    ) -> dict[str, any]:
        """Execute processing pipeline"""

        words = []
        boxes = []
        documents = docs_from_image(frames)
        assert len(documents) == len(frames)

        for page_idx in range(len(frames)):
            page_words, page_boxes = get_words_and_boxes(ocr_results, page_idx)
            words.append(page_words)
            boxes.append(page_boxes)

        assert len(words) == len(boxes)

        context = PipelineContext(pipeline_id="classification_pipeline")
        context["metadata"] = {}

        for pipe in processing_pipeline:
            try:
                # create a PipelineContext and pass it to the component
                pipe_results = pipe.run(documents, context, words=words, boxes=boxes)
                if pipe_results.state is not None:
                    if not isinstance(pipe_results.state, DocList):
                        raise ValueError(
                            f"Invalid state type : {type(pipe_results.state)}"
                        )
                    documents = pipe_results.state
            except Exception as e:
                if not self.silence_exceptions:
                    raise ValueError("Error executing pipe") from e
                self.logger.error(f"Error executing pipe : {e}")

        # TODO : This is temporary, we need to make this configurable
        self.logger.info("### ClassificationPipeline results")
        self.logger.info(context["metadata"])

        page_indexer_meta = (
            context["metadata"]["page_indexer"]
            if "page_indexer" in context["metadata"]
            else []
        )
        page_classifier_meta = (
            context["metadata"]["page_classifier"]
            if "page_classifier" in context["metadata"]
            else []
        )

        for idx, page_result in enumerate(page_classifier_meta):
            for detail in page_result["details"]:
                page = int(detail["page"])
                classification = detail["classification"]
                filtered_classifiers = {}

                for key, val in sub_classifiers.items():
                    fileter_config = val["filter"]
                    filter_type = fileter_config["type"]
                    filter_pattern = fileter_config["pattern"]

                    if filter_type == "exact" and classification == filter_pattern:
                        self.logger.info(f"Adding sub-classifier : {key}")
                        filtered_classifiers[key] = val

                if filtered_classifiers:
                    self.logger.info(
                        f"Filtered classifiers : {filtered_classifiers.keys()}"
                    )
                    sub_classifier_pipeline = ClassifierPipelineComponent(
                        name="sub_classifier_pipeline",
                        document_classifiers=filtered_classifiers,
                    )

                    ctx = PipelineContext(pipeline_id="sub_classification_pipeline")
                    ctx["metadata"] = {}
                    pipe_results = sub_classifier_pipeline.run(
                        documents[page : page + 1],
                        ctx,
                        words=[words[page]],
                        boxes=[boxes[page]],
                    )
                    detail["sub_classifier"] = ctx["metadata"]["page_classifier"]

        # TODO : Read from config
        # Classification strategy: max_score, max_votes, max_score_with_diff
        prediction_agent = "majority"
        tie_break_policy = "best_with_diff"
        voter = get_voting_strategy(prediction_agent, tie_break_policy, max_diff=0.25)

        class_by_page = self.group_results_by_page("classifier", page_classifier_meta)
        score_by_page = {}
        for page, details in class_by_page.items():
            score_by_page[page] = voter([ClassificationResult(**x) for x in details])

        classifier_results = {
            "strategy": prediction_agent,
            "tie_break_policy": tie_break_policy,
            "pages": {},
        }

        for page in list(class_by_page.keys()):
            classifier_results["pages"][page] = {
                "details": class_by_page[page],
                "best": score_by_page[page],
            }

        # Indexer results
        indexer_by_page = self.group_results_by_page("indexer", page_indexer_meta)
        indexer_results = {"strategy": "default", "pages": {}}

        for page in list(indexer_by_page.keys()):
            indexer_results["pages"][page] = {"details": indexer_by_page[page]}

        return {"classifier": classifier_results, "indexer": indexer_results}

    def group_results_by_page(
        self, group_key: str, page_meta: List[dict[str, any]]
    ) -> dict:
        """Group the results by page"""
        group_by_page = {}
        for idx, page_result in enumerate(page_meta):
            indexer = page_result[group_key]
            for detail in page_result["details"]:
                page = int(detail["page"])
                if page not in group_by_page:
                    group_by_page[page] = []
                detail[group_key] = indexer
                group_by_page[page].append(detail)

        return group_by_page
