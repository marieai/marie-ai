# from typing import Optional
#
# from marie.logging.logger import MarieLogger
#
#
#
# class JobManager:
#     """Provide python APIs for job submission and management.
#
#     It does not provide persistence, all info will be lost if the cluster
#     goes down.
#     """
#
#     def __init__(
#         self,
#     ):
#         self.logger = MarieLogger("JobManager")
#         # self._log_client = JobLogStorageClient()
#         self.monitored_jobs = set()
#
#     async def submit_job(
#         self,
#         *,
#         entrypoint: str,
#     ) -> str:
#         """
#         Job execution happens asynchronously.
#         """
#
#         self.logger.info(f"Starting job with submission_id: {submission_id}")
#         job_info = JobInfo(
#             entrypoint=entrypoint,
#             status=JobStatus.PENDING,
#             start_time=int(time.time() * 1000),
#             metadata=metadata,
#             runtime_env=runtime_env,
#             entrypoint_num_cpus=entrypoint_num_cpus,
#             entrypoint_num_gpus=entrypoint_num_gpus,
#             entrypoint_resources=entrypoint_resources,
#         )
#         new_key_added = await self._job_info_client.put_info(
#             submission_id, job_info, overwrite=False
#         )
#         if not new_key_added:
#             raise ValueError(
#                 f"Job with submission_id {submission_id} already exists. "
#                 "Please use a different submission_id."
#             )
#
#         # Wait for the actor to start up asynchronously so this call always
#         # returns immediately and we can catch errors with the actor starting
#         # up.
#         try:
#             resources_specified = any(
#                 [
#                     entrypoint_num_cpus is not None and entrypoint_num_cpus > 0,
#                     entrypoint_num_gpus is not None and entrypoint_num_gpus > 0,
#                     entrypoint_resources not in [None, {}],
#                 ]
#             )
#             scheduling_strategy = await self._get_scheduling_strategy(
#                 resources_specified
#             )
#             if self.event_logger:
#                 self.event_logger.info(
#                     f"Started a ray job {submission_id}.", submission_id=submission_id
#                 )
#             supervisor = self._supervisor_actor_cls.options(
#                 lifetime="detached",
#                 name=JOB_ACTOR_NAME_TEMPLATE.format(job_id=submission_id),
#                 num_cpus=entrypoint_num_cpus,
#                 num_gpus=entrypoint_num_gpus,
#                 resources=entrypoint_resources,
#                 scheduling_strategy=scheduling_strategy,
#                 runtime_env=self._get_supervisor_runtime_env(
#                     runtime_env, resources_specified
#                 ),
#                 namespace=SUPERVISOR_ACTOR_RAY_NAMESPACE,
#             ).remote(submission_id, entrypoint, metadata or {}, self._gcs_address)
#             supervisor.run.remote(
#                 _start_signal_actor=_start_signal_actor,
#                 resources_specified=resources_specified,
#             )
#
#             # Monitor the job in the background so we can detect errors without
#             # requiring a client to poll.
#             run_background_task(
#                 self._monitor_job(submission_id, job_supervisor=supervisor)
#             )
#         except Exception as e:
#             await self._job_info_client.put_status(
#                 submission_id,
#                 JobStatus.FAILED,
#                 message=f"Failed to start Job Supervisor actor: {e}.",
#             )
#
#         return submission_id
#
#     def stop_job(self, job_id) -> bool:
#         """Request a job to exit, fire and forget.
#
#         Returns whether or not the job was running.
#         """
#         job_supervisor_actor = self._get_actor_for_job(job_id)
#         if job_supervisor_actor is not None:
#             # Actor is still alive, signal it to stop the driver, fire and
#             # forget
#             job_supervisor_actor.stop.remote()
#             return True
#         else:
#             return False
#
#     async def delete_job(self, job_id):
#         """Delete a job's info and metadata from the cluster."""
#         job_status = await self._job_info_client.get_status(job_id)
#
#         if job_status is None or not job_status.is_terminal():
#             raise RuntimeError(
#                 f"Attempted to delete job '{job_id}', "
#                 f"but it is in a non-terminal state {job_status}."
#             )
#
#         await self._job_info_client.delete_info(job_id)
#         return True
