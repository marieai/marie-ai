from typing import List

from marie.job.pydantic_models import JobPartition


class JobSplitter:
    @staticmethod
    def calculate_partitions(page_count: int, chunk_size: int) -> List[JobPartition]:
        partitions = []

        chunks = page_count // chunk_size
        remainder = page_count % chunk_size

        end_index = page_count
        chunk_index = 0

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
