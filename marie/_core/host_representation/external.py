from multiprocessing import RLock
from typing import Optional, Callable

from marie import check
from marie._core.host_representation.job_index import JobIndex
from marie._core.host_representation.represented import RepresentedJob


class ExternalJobData:
    pass


class RepositoryHandle:
    pass


class ExternalJobRef:
    pass


class JobHandle:
    pass


class ExternalJob(RepresentedJob):
    """ExternalJob is a object that represents a loaded job definition that
    is resident in another process or container. Host processes such as dagit use
    objects such as these to interact with user-defined artifacts.
    """

    def __init__(
        self,
        external_job_data: Optional[ExternalJobData],
        repository_handle: RepositoryHandle,
        external_job_ref: Optional[ExternalJobRef] = None,
        ref_to_data_fn: Optional[Callable[[ExternalJobRef], ExternalJobData]] = None,
    ):
        check.inst_param(repository_handle, "repository_handle", RepositoryHandle)
        check.opt_inst_param(external_job_data, "external_job_data", ExternalJobData)

        self._repository_handle = repository_handle

        self._memo_lock = RLock()
        self._index: Optional[JobIndex] = None

        self._data = external_job_data
        self._ref = external_job_ref
        self._ref_to_data_fn = ref_to_data_fn

        if external_job_data:
            self._active_preset_dict = {
                ap.name: ap for ap in external_job_data.active_presets
            }
            self._name = external_job_data.name
            self._snapshot_id = self._job_index.job_snapshot_id

        elif external_job_ref:
            self._active_preset_dict = {
                ap.name: ap for ap in external_job_ref.active_presets
            }
            self._name = external_job_ref.name
            if ref_to_data_fn is None:
                check.failed(
                    "ref_to_data_fn must be passed when using deferred snapshots"
                )
            self._snapshot_id = external_job_ref.snapshot_id
        else:
            check.failed("Expected either job data or ref, got neither")

        self._handle = JobHandle(self._name, repository_handle)
