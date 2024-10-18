from abc import ABC, abstractmethod
from typing import List

from marie.job.pydantic_models import JobPartition
from marie.scheduler.models import WorkInfo


class JobPartitioner(ABC):
    """
    Abstract base class for partitioning and aggregating work items.
    """

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    @abstractmethod
    def partition(self, work_info: WorkInfo) -> List[JobPartition]:
        """
        Partition the work_info data into smaller chunks.

        :param work_info: The work item to be partitioned.
        :return: A list of JobPartition objects.
        """
        pass

    @abstractmethod
    def aggregate(self, results: List[WorkInfo]) -> WorkInfo:
        """
        Aggregate the results from the partitions.

        :param results: The list of results from the partitions.
        :return: The aggregated result.
        """
        pass

    @staticmethod
    def calculate_partitions(page_count: int, chunk_size: int) -> List[JobPartition]:
        """
        Calculate the partitions for a given page count based on the chunk size.

        This method divides the total number of pages into smaller chunks (partitions)
        based on the specified chunk size. Each partition is represented as a JobPartition
        object with a unique chunk index, start index, and end index.

        :param page_count: The total number of pages to be partitioned.
        :param chunk_size: The size of each chunk (number of pages per partition).
        :return: A list of JobPartition objects representing the partitions.
        """
        partitions = []

        # Calculate the number of full chunks and the remainder
        chunks = page_count // chunk_size
        remainder = page_count % chunk_size

        end_index = page_count
        chunk_index = 0

        # Create partitions for full chunks
        for i in range(chunks):
            start_index = i * chunk_size
            end_index = start_index + chunk_size

            partitions.append(
                JobPartition(
                    chunk_index=chunk_index,
                    start_index=start_index,
                    end_index=end_index,
                )
            )
            chunk_index += 1

        if remainder > 0:
            start_index = 0 if chunks == 0 else end_index
            end_index = start_index + remainder

            partitions.append(
                JobPartition(
                    chunk_index=chunk_index,
                    start_index=start_index,
                    end_index=end_index,
                )
            )

        return partitions

    @staticmethod
    def can_split(page_count: int, chunk_size: int) -> bool:
        return page_count > chunk_size
