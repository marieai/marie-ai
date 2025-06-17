from abc import ABC
from collections import defaultdict
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
            self.logger.info(f"Processing classifier: {pipeline_name}/{group}")
            (document_classifiers, sub_classifiers) = self.build_classifier_component(
                classifier_group, group
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
                processing_pipeline,
                sub_classifiers,
                frames,
                ocr_results,
                "classification_pipeline",
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

    @staticmethod
    def build_classifier_component(
        classifier_group, group
    ) -> (ClassifierPipelineComponent, dict):
        document_classifiers = classifier_group["classifiers"]
        sub_classifiers = (
            classifier_group["sub_classifiers"]
            if "sub_classifiers" in classifier_group
            else dict()
        )

        return (
            ClassifierPipelineComponent(
                name=f"{group}_classifier_component",
                document_classifiers=document_classifiers,
            ),
            sub_classifiers,
        )

    def execute_pipeline(
        self,
        processing_pipeline: List[PipelineComponent],
        sub_classifiers: dict[str, any],
        frames: List,
        ocr_results: dict,
        pipeline_id: str = "default_pipeline",
        include_ocr_lines: bool = False,
    ) -> dict[str, any]:
        """Execute processing pipeline"""

        words = []
        boxes = []
        lines = []
        documents = docs_from_image(frames)
        assert len(documents) == len(frames)

        for page_idx in range(len(frames)):
            page_words, page_boxes, page_lines = get_words_and_boxes(
                ocr_results, page_idx, include_lines=True
            )
            words.append(page_words)
            boxes.append(page_boxes)
            if include_ocr_lines:
                lines.append(page_lines)

        assert len(words) == len(boxes)
        if include_ocr_lines:
            assert len(words) == len(lines)

        context = PipelineContext(pipeline_id=pipeline_id)
        context["metadata"] = {}

        for pipe in processing_pipeline:
            try:
                # create a PipelineContext and pass it to the component
                pipe_results = pipe.run(
                    documents, context, words=words, boxes=boxes, lines=lines
                )
                if pipe_results.state is not None:
                    if not isinstance(pipe_results.state, DocList):
                        raise ValueError(
                            f"Invalid state type : {type(pipe_results.state)}"
                        )
            except Exception as e:
                if not self.silence_exceptions:
                    raise ValueError("Error executing pipe") from e
                self.logger.error(f"Error executing pipe : {e}")

        self.logger.info(f"### {pipeline_id} results")
        self.logger.debug(context["metadata"])

        pipeline_meta = dict()
        for key in context["metadata"]:
            key_meta = context["metadata"][key]
            if key == "page_classifier":
                self.run_sub_classifier_pipeline(
                    key_meta,
                    sub_classifiers,
                    words,
                    boxes,
                    documents,
                    pipeline_id=pipeline_id,
                )
                pipeline_meta["classifiers"] = self.get_classifier_results(key_meta)
            if key == "page_indexer":
                pipeline_meta["indexes"] = self.get_indexer_results(key_meta)

        return pipeline_meta

    def run_sub_classifier_pipeline(
        self,
        page_classifier_meta,
        sub_classifiers: dict[str, any],
        words: List,
        boxes: List,
        documents: DocList,
        pipeline_id="default_pipeline",
    ):
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

                    ctx = PipelineContext(pipeline_id=f"sub_{pipeline_id}")
                    ctx["metadata"] = {}
                    sub_classifier_pipeline.run(
                        documents[page : page + 1],
                        ctx,
                        words=[words[page]],
                        boxes=[boxes[page]],
                    )
                    detail["sub_classifier"] = ctx["metadata"]["page_classifier"]

        return page_classifier_meta

    def get_classifier_results(
        self,
        page_classifier_meta,
        prediction_agent="max_score",
        tie_break_policy="best",
    ):

        # TODO : Read from config
        # Classification strategy: max_score, max_votes, max_score_with_diff
        classifier_results = {
            "strategy": prediction_agent,
            "tie_break_policy": tie_break_policy,
            "pages": {},
        }

        voter = get_voting_strategy(prediction_agent, tie_break_policy, max_diff=0.25)

        class_by_page = self.group_results_by_page("classifier", page_classifier_meta)
        score_by_page = {}
        for page, details in class_by_page.items():
            score_by_page[page] = voter([ClassificationResult(**x) for x in details])

        for page in list(class_by_page.keys()):
            classifier_results["pages"][page] = {
                "details": class_by_page[page],
                "best": score_by_page[page],
            }

        return classifier_results

    def get_indexer_results(self, page_indexer_meta, strategy="default"):
        if strategy != "default":
            raise NotImplementedError(
                f"Strategy '{strategy}' not implemented for indexing results"
            )
        # TODO: add other indexing strategies

        results = defaultdict(
            lambda: {
                "strategy": strategy,
                "pages": defaultdict(lambda: {"details": [], "best": None}),
            }
        )
        for indexing_meta in page_indexer_meta:
            indexing_task = indexing_meta["indexing"]
            results[indexing_task]["task"] = indexing_task
            for detail in indexing_meta["details"]:
                page = str(detail["page"])
                page_index = results[indexing_task]["pages"][page]
                page_index["details"].append(detail)

                # default strategy is just take the first for now
                if page_index["best"] is None and strategy == "default":
                    page_index["best"] = detail

        return results

    def group_results_by_page(
        self, group_key: str, page_meta: List[dict[str, any]]
    ) -> dict:
        """Group the results by page"""
        group_by_page = defaultdict(list)
        for idx, page_result in enumerate(page_meta):
            indexer = page_result[group_key]
            for detail in page_result["details"]:
                detail[group_key] = indexer
                page = int(detail["page"])
                group_by_page[page].append(detail)

        return group_by_page
