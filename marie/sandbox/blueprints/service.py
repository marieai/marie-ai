"""Blueprint import service for the Marie sandbox gateway.

Translates a parsed blueprint manifest into per-artifact installs against
THIS gateway's own registries and storage.  Where a capability does not yet
exist in marie-ai (dify-parity gaps, missing config-repo mounts, etc.) the
artifact is recorded as ``deferred`` with an explicit reason instead of being
silently skipped.

Artifact kind dispatch table
-----------------------------
kind            | status   | mechanism
----------------|----------|--------------------------------------------------
query_plan      | applied  | QueryPlanRegistry.register_from_json() (in-memory)
sample_data     | applied  | Acknowledged ref (no persistent storage needed)
tenant_registry | deferred | Sandbox org/workspace seeded in Wave-1 only;
                |          | per-document tenant schema requires dify-parity
                |          | storage backend
rag_index       | deferred | No RAG backend in sandbox schema; pending
                |          | dify-parity weaviate/qdrant integration
rag_source      | deferred | Depends on rag_index; deferred with it
prompt_package  | deferred | Config-repo (git-backed prompt repo) not mounted
                |          | in sandbox; pending config-repo injection
script_package  | deferred | Config-repo not mounted in sandbox
webapp_package  | deferred | Webapp runner registry not available in sandbox
prefab          | deferred | Installer not wired in blueprint-import-service
skill           | deferred | Installer not wired in blueprint-import-service
agent           | deferred | Installer not wired in blueprint-import-service
"""

from __future__ import annotations

from typing import Any

from marie.logging_core.logger import MarieLogger
from marie.query_planner.base import QueryPlanRegistry
from marie.sandbox.blueprints.models import ArtifactResult, BlueprintImportResponse

_logger = MarieLogger('marie.sandbox.blueprints.service')

# Reason strings for fully-deferred kinds — explicit so tests can assert them.
_DEFERRED_REASONS: dict[str, str] = {
    'tenant_registry': (
        'sandbox org/workspace already seeded in Wave-1 (seed_defaults); '
        'per-document tenant schema requires dify-parity storage backend'
    ),
    'rag_index': (
        'RAG index storage not available in sandbox schema; '
        'pending dify-parity weaviate/qdrant backend integration'
    ),
    'rag_source': (
        'RAG source depends on rag_index which is deferred; '
        'pending dify-parity rag-backend work'
    ),
    'prompt_package': (
        'prompt repository (git-backed config repo) not mounted in sandbox; '
        'pending config-repo sandbox injection'
    ),
    'script_package': (
        'script repository not mounted in sandbox; '
        'pending config-repo sandbox injection'
    ),
    'webapp_package': (
        'webapp runner registry not available in sandbox; '
        'pending webapp deployment work'
    ),
    'prefab': 'installer not wired in blueprint-import-service or marie-ai',
    'skill': 'installer not wired in blueprint-import-service or marie-ai',
    'agent': 'installer not wired in blueprint-import-service or marie-ai',
}


class BlueprintImportService:
    """Install a blueprint manifest into this gateway's own registries.

    Designed to be called once per sandbox seeding cycle.  All operations are
    idempotent: calling :meth:`import_blueprint` twice with the same manifest
    produces the same end-state.
    """

    def import_blueprint(
        self, blueprint_id: str, manifest: dict[str, Any]
    ) -> BlueprintImportResponse:
        """Process every artifact in *manifest* and return a partial result.

        Args:
            blueprint_id: Identifier as received from the Studio seam.
            manifest:     Parsed blueprint.yaml dict.

        Returns:
            :class:`BlueprintImportResponse` with per-artifact outcomes.
        """
        artifacts = manifest.get('artifacts') or []
        if not isinstance(artifacts, list):
            return BlueprintImportResponse(
                blueprint_id=blueprint_id,
                status='failed',
                message='manifest.artifacts is not a list',
            )

        applied: list[ArtifactResult] = []
        deferred: list[ArtifactResult] = []
        failed: list[ArtifactResult] = []

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            result = self._dispatch(artifact)
            if result.status == 'applied':
                applied.append(result)
            elif result.status == 'deferred':
                deferred.append(result)
            else:
                failed.append(result)

        if failed and not applied and not deferred:
            overall = 'failed'
        elif deferred or failed:
            overall = 'partial'
        else:
            overall = 'completed'

        _logger.info(
            f'Blueprint {blueprint_id!r} import: '
            f'{len(applied)} applied, {len(deferred)} deferred, {len(failed)} failed '
            f'→ {overall}'
        )
        return BlueprintImportResponse(
            blueprint_id=blueprint_id,
            status=overall,
            applied=applied,
            deferred=deferred,
            failed=failed,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, artifact: dict[str, Any]) -> ArtifactResult:
        kind = str(artifact.get('kind', ''))
        ref = str(artifact.get('ref', kind))

        if kind == 'query_plan':
            return self._install_query_plan(artifact)
        if kind == 'sample_data':
            return self._install_sample_data(artifact)
        if kind in _DEFERRED_REASONS:
            return ArtifactResult(
                ref=ref,
                kind=kind,
                status='deferred',
                reason=_DEFERRED_REASONS[kind],
            )

        reason = f'unknown artifact kind {kind!r}; cannot install'
        _logger.warning(f'Artifact {ref!r}: {reason}')
        return ArtifactResult(ref=ref, kind=kind, status='deferred', reason=reason)

    # ------------------------------------------------------------------
    # Concrete installers
    # ------------------------------------------------------------------

    def _install_query_plan(self, artifact: dict[str, Any]) -> ArtifactResult:
        ref = str(artifact.get('ref', 'query_plan'))
        create = artifact.get('create') or {}
        if not isinstance(create, dict):
            return ArtifactResult(
                ref=ref,
                kind='query_plan',
                status='deferred',
                reason='artifact.create is missing or not a mapping',
            )

        name: str = str(create.get('name') or ref)
        plan_definition = create.get('planDefinition') or create.get('plan_definition')

        if plan_definition is None:
            # planDefinitionPath is archive-relative and not accessible in the gateway.
            return ArtifactResult(
                ref=ref,
                kind='query_plan',
                status='deferred',
                reason=(
                    'no inline planDefinition in manifest; '
                    'archive-relative planDefinitionPath is not accessible in the gateway context'
                ),
            )

        # Idempotent: already registered → report applied without re-registering.
        if QueryPlanRegistry.get_metadata(name) is not None:
            _logger.info(
                f'Query plan {name!r} already registered (idempotent re-import)'
            )
            return ArtifactResult(
                ref=ref,
                kind='query_plan',
                status='applied',
                reason='already registered (idempotent)',
            )

        success = QueryPlanRegistry.register_from_json(
            name=name,
            plan_definition=plan_definition,
            description=create.get('description') or None,
            version=str(create.get('version') or '1.0.0'),
            tags=list(create.get('tags') or []),
            category=create.get('category') or None,
        )

        if success:
            _logger.info(
                f'Registered query plan {name!r} from blueprint artifact {ref!r}'
            )
            return ArtifactResult(ref=ref, kind='query_plan', status='applied')

        return ArtifactResult(
            ref=ref,
            kind='query_plan',
            status='failed',
            reason='QueryPlanRegistry.register_from_json returned False (see gateway logs)',
        )

    @staticmethod
    def _install_sample_data(artifact: dict[str, Any]) -> ArtifactResult:
        ref = str(artifact.get('ref', 'sample_data'))
        # sample_data has no persistent storage requirement in the gateway; acknowledge the ref.
        return ArtifactResult(
            ref=ref,
            kind='sample_data',
            status='applied',
            reason='acknowledged in memory (no persistent storage required)',
        )
