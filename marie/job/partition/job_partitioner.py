from typing import List

from marie.api import parse_payload_to_docs_sync
from marie.job.partition.base import JobPartitioner
from marie.job.pydantic_models import JobPartition
from marie.scheduler.models import WorkInfo


class MarieJobPartitioner(JobPartitioner):
    """
    MarieJobPartitioner class is used for partitioning and aggregating work items
    """

    def __init__(self, chunk_size: int):
        super().__init__(chunk_size)

    def aggregate(self, results: List[WorkInfo]) -> WorkInfo:
        pass

    def partition(self, work_info: WorkInfo) -> List[JobPartition]:
        partitions = []
        metadata = work_info.data.get("metadata", {})
        parameters, asset_doc = parse_payload_to_docs_sync(
            metadata, clear_payload=False
        )
        # print(f"Partitioning document : {asset_doc.asset_key}")
        return partitions
