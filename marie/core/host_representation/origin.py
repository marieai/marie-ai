from abc import abstractmethod, ABC
from typing import NamedTuple, Mapping, Any

from marie import check
from marie.core.definitions.selector import RepositorySelector, PartitionSetSelector
from marie.serdes import create_snapshot_id


class CodeLocationOrigin(ABC, tuple):
    """Serializable representation of a CodeLocation that can be used to
    uniquely identify the location or reload it in across process boundaries.
    """

    @property
    def is_reload_supported(self) -> bool:
        return True

    @property
    def is_shutdown_supported(self) -> bool:
        return False

    def shutdown_server(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_display_metadata(self) -> Mapping[str, Any]:
        pass

    def get_id(self) -> str:
        # Type-ignored because `create_snapshot` takes a `NamedTuple`, and all descendants of this
        # class are `NamedTuple`, but we can't specify `NamedTuple` in the signature here.
        return create_snapshot_id(self)  # type: ignore

    @property
    @abstractmethod
    def location_name(self) -> str:
        pass

    @abstractmethod
    def create_location(self) -> "CodeLocation":
        pass


class ExternalRepositoryOrigin(
    NamedTuple(
        "_ExternalRepositoryOrigin",
        [("code_location_origin", CodeLocationOrigin), ("repository_name", str)],
    )
):
    """Serializable representation of an ExternalRepository that can be used to
    uniquely it or reload it in across process boundaries.
    """

    def __new__(cls, code_location_origin: CodeLocationOrigin, repository_name: str):
        return super(ExternalRepositoryOrigin, cls).__new__(
            cls,
            check.inst_param(
                code_location_origin, "code_location_origin", CodeLocationOrigin
            ),
            check.str_param(repository_name, "repository_name"),
        )

    def get_id(self) -> str:
        return create_snapshot_id(self)

    def get_selector_id(self) -> str:
        return create_snapshot_id(
            RepositorySelector(
                self.code_location_origin.location_name, self.repository_name
            )
        )

    def get_label(self) -> str:
        return f"{self.repository_name}@{self.code_location_origin.location_name}"

    def get_job_origin(self, job_name: str) -> "ExternalJobOrigin":
        return ExternalJobOrigin(self, job_name)

    def get_instigator_origin(self, instigator_name: str) -> "ExternalInstigatorOrigin":
        return ExternalInstigatorOrigin(self, instigator_name)

    def get_partition_set_origin(
        self, partition_set_name: str
    ) -> "ExternalPartitionSetOrigin":
        return ExternalPartitionSetOrigin(self, partition_set_name)


class ExternalJobOrigin(
    NamedTuple(
        "_ExternalJobOrigin",
        [("external_repository_origin", ExternalRepositoryOrigin), ("job_name", str)],
    )
):
    """Serializable representation of an ExternalPipeline that can be used to
    uniquely it or reload it in across process boundaries.
    """

    def __new__(
        cls, external_repository_origin: ExternalRepositoryOrigin, job_name: str
    ):
        return super(ExternalJobOrigin, cls).__new__(
            cls,
            check.inst_param(
                external_repository_origin,
                "external_repository_origin",
                ExternalRepositoryOrigin,
            ),
            check.str_param(job_name, "job_name"),
        )

    def get_id(self) -> str:
        return create_snapshot_id(self)

    @property
    def location_name(self) -> str:
        return self.external_repository_origin.code_location_origin.location_name


class ExternalInstigatorOrigin(
    NamedTuple(
        "_ExternalInstigatorOrigin",
        [
            ("external_repository_origin", ExternalRepositoryOrigin),
            ("instigator_name", str),
        ],
    )
):
    """Serializable representation of an ExternalJob that can be used to
    uniquely it or reload it in across process boundaries.
    """

    def __new__(
        cls, external_repository_origin: ExternalRepositoryOrigin, instigator_name: str
    ):
        return super(ExternalInstigatorOrigin, cls).__new__(
            cls,
            check.inst_param(
                external_repository_origin,
                "external_repository_origin",
                ExternalRepositoryOrigin,
            ),
            check.str_param(instigator_name, "instigator_name"),
        )

    def get_id(self) -> str:
        return create_snapshot_id(self)


class ExternalPartitionSetOrigin(
    NamedTuple(
        "_PartitionSetOrigin",
        [
            ("external_repository_origin", ExternalRepositoryOrigin),
            ("partition_set_name", str),
        ],
    )
):
    """Serializable representation of an ExternalPartitionSet that can be used to
    uniquely it or reload it in across process boundaries.
    """

    def __new__(
        cls,
        external_repository_origin: ExternalRepositoryOrigin,
        partition_set_name: str,
    ):
        return super(ExternalPartitionSetOrigin, cls).__new__(
            cls,
            check.inst_param(
                external_repository_origin,
                "external_repository_origin",
                ExternalRepositoryOrigin,
            ),
            check.str_param(partition_set_name, "partition_set_name"),
        )

    def get_id(self) -> str:
        return create_snapshot_id(self)

    @property
    def selector(self) -> PartitionSetSelector:
        return PartitionSetSelector(
            self.external_repository_origin.code_location_origin.location_name,
            self.external_repository_origin.repository_name,
            self.partition_set_name,
        )

    def get_selector_id(self) -> str:
        return create_snapshot_id(self.selector)
