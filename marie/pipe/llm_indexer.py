from collections import defaultdict
from typing import List, Optional

from docarray import DocList

from marie.api.docs import DOC_KEY_INDEXER
from marie.logging_core.logger import MarieLogger
from marie.pipe.base import PipelineComponent, PipelineContext, PipelineResult


class LLMIndexerPipelineComponent(PipelineComponent):
    def __init__(
        self,
        name: str,
        document_indexers: dict,
        llm_tasks: list,
        logger: MarieLogger = None,
        silence_exceptions: bool = False,
    ) -> None:
        """
        Initialize the IndexerPipelineComponent.

        :param name: The name of the pipeline component.
        :param document_indexers: A dictionary containing document indexers.
        :param logger: An optional logger for logging messages.
        :param silence_exceptions: Whether to suppress exceptions.
        """
        super().__init__(name, logger=logger)
        self.silence_exceptions = silence_exceptions
        if not isinstance(document_indexers, dict):
            raise ValueError("document_indexers must be a dictionary")
        self.document_indexers = document_indexers
        self.llm_tasks = llm_tasks

    def predict(
        self,
        documents: DocList,
        context: Optional[PipelineContext] = None,
        *,
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
        lines: List[List[int]] = None,
    ) -> PipelineResult:
        """
        Predict the named entities in the documents.

        :param documents: The input documents.
        :param context: The pipeline context.
        :param words: The list of words.
        :param boxes: The list of boxes.
        :param lines: The list of lines.
        :return: The pipeline result.
        """
        context["metadata"]["page_indexer"] = self.index_documents(
            documents,
            words,
            boxes,
            lines,
        )

        return PipelineResult(documents)

    def index_documents(
        self,
        documents: DocList,
        words: List[List[str]] = None,
        boxes: List[List[List[int]]] = None,
        lines: List[List[int]] = None,
    ) -> List[dict]:
        """
        Extract indexes from the documents.

        :param documents: The documents to extract index data from.
        :param words: The list of words.
        :param boxes: The list of boxes.
        :param lines: The list of lines.
        :return: The extracted named entities.
        """

        document_meta = []
        for key, document_indexer in self.document_indexers.items():
            self.logger.info(f"Indexing document : {key}")
            indexer = document_indexer["indexer"]
            group = document_indexer["group"]

            # TODO: filtering by page level classifications
            # has_filter = "filter" in document_indexer
            try:
                results = indexer.run(
                    documents=documents,
                    words=words,
                    boxes=boxes,
                    lines=lines,
                    tasks=self.llm_tasks,
                )

                task_meta = defaultdict(list)
                for idx, document in enumerate(results):
                    assert DOC_KEY_INDEXER in document.tags
                    assert all(
                        task_name in document.tags[DOC_KEY_INDEXER]
                        for task_name in self.llm_tasks
                    )

                    indexed_values_by_task = document.tags[DOC_KEY_INDEXER]
                    for task_name, (
                        indexed_values,
                        error_data,
                    ) in indexed_values_by_task.items():
                        if task_name not in self.llm_tasks:
                            continue

                        page_task_meta = {
                            "page": str(idx),
                            "indexing": indexed_values,
                            "indexer": key,
                        }
                        if error_data:
                            page_task_meta["error"] = error_data
                        task_meta[task_name].append(page_task_meta)

                for task_name, meta in task_meta.items():
                    document_meta.append(
                        {
                            "indexing": task_name,
                            "indexer": key,
                            "details": meta,
                        }
                    )

            except Exception as e:
                if not self.silence_exceptions:
                    raise ValueError("Error indexing document") from e
                self.logger.warning("Error indexing document : ", exc_info=1)
                document_meta.append(
                    {
                        "indexing": None,
                        "indexer": key,
                        "details": [],
                    }
                )

        return document_meta
