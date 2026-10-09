"""Marie AI persistence adapter for reusable LLM scheduler configuration."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional

import psycopg
from marie.engine.llm_queue.admission_policy import AdmissionPolicy
from psycopg.abc import RV, PQGen
from psycopg.types.json import Jsonb

from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.marie.dispatch_policy import (
    build_policy_generation_snapshot,
    persisted_dispatch_policy,
    validate_dispatch_policy,
)
from marie.storage.database.postgres import PostgresqlMixin

DEFAULT_SCHEDULER_CONFIG_SCHEMA = "marie_scheduler"
DEFAULT_FABRIC_CONFIG_TABLE = "llm_queue_fabric_config"
DEFAULT_POOL_TABLE = "llm_queue_pool"
MAX_DATABASE_LANES = 1000
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ACTOR_ID_RE = re.compile(r'^[A-Za-z0-9_.@-]{1,128}$')


class _PolicyConnection(psycopg.Connection):
    policy_read_deadline: float | None = None

    def wait(self, gen: PQGen[RV], interval: float = 0.1) -> RV:
        deadline = min(
            time.monotonic() + 3.0, self.policy_read_deadline or float('inf')
        )

        def bounded() -> PQGen[RV]:
            try:
                state = next(gen)
                while True:
                    if time.monotonic() >= deadline:
                        self.close()
                        raise psycopg.OperationalError('Policy database I/O timed out')
                    ready = yield state
                    state = gen.send(ready)
            except StopIteration as done:
                return done.value
            finally:
                gen.close()

        return super().wait(bounded(), interval=min(interval, 0.05))


@dataclass(frozen=True, slots=True)
class ActivatedPolicy:
    fabric_group_id: str
    generation: int
    policy_digest: str
    activated_by: str
    created_on: datetime


class PostgresSchedulerConfigRepository(PostgresqlMixin):
    """Load engine scheduler configuration from Marie's PostgreSQL store."""

    connection_class = _PolicyConnection

    def __init__(
        self,
        config: dict[str, Any],
        *,
        logger: Optional[MarieLogger] = None,
    ) -> None:
        super().__init__()
        self.logger = logger or MarieLogger(self.__class__.__name__)
        self.config_schema = _sql_identifier(
            config.get("schema") or DEFAULT_SCHEDULER_CONFIG_SCHEMA,
            label="schema",
        )
        self._setup_storage(
            {**config, "pool_acquire_timeout_seconds": 1.0}, connection_only=True
        )

    def load_scheduler_config(self, fabric_group_id: str) -> dict[str, Any]:
        fabric_config_table = f"{self.config_schema}.{DEFAULT_FABRIC_CONFIG_TABLE}"
        pool_table = f"{self.config_schema}.{DEFAULT_POOL_TABLE}"
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            cursor.execute(
                f"""
                SELECT policy, total_concurrent_dispatch, metadata
                FROM {fabric_config_table}
                WHERE fabric_group_id = %s
                  AND enabled = true
                """,
                (fabric_group_id,),
            )
            fabric_config = cursor.fetchone()
            if fabric_config is None:
                raise ValueError(
                    f"LLM queue Runtime Fabric group {fabric_group_id!r} is not configured"
                )

            cursor.execute(
                f"""
                SELECT
                    pool_id,
                    display_name,
                    endpoint_url,
                    quantum,
                    min_concurrent,
                    max_concurrent,
                    max_burst_per_visit,
                    enabled,
                    metadata
                FROM {pool_table}
                WHERE fabric_group_id = %s
                ORDER BY sort_order ASC, pool_id ASC
                LIMIT {MAX_DATABASE_LANES + 1}
                """,
                (fabric_group_id,),
            )
            rows = cursor.fetchall()
            if len(rows) > MAX_DATABASE_LANES:
                raise ValueError("LLM queue database policy exceeds 1000 lanes")
            lanes = [
                {
                    "pool_id": row[0],
                    "display_name": row[1],
                    "endpoint_url": row[2],
                    "quantum": row[3],
                    "min_concurrent": row[4],
                    "max_concurrent": row[5],
                    "max_burst_per_visit": row[6],
                    "enabled": row[7],
                    "metadata": row[8],
                }
                for row in rows
            ]
            conn.commit()
            return {
                "policy": fabric_config[0],
                "total_concurrent_dispatch": fabric_config[1],
                "lanes": lanes,
                "metadata": fabric_config[2],
            }
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def activate_admission_policy(
        self, fabric_group_id: str, actor_id: str
    ) -> ActivatedPolicy:
        if not _ACTOR_ID_RE.fullmatch(actor_id):
            raise ValueError('Invalid routing policy actor')
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '3000ms'")
            data = self._load_desired_policy(cursor, fabric_group_id, lock=True)
            cursor.execute(
                f"""
                SELECT COALESCE(MAX(generation), 0)
                FROM {self.config_schema}.llm_queue_policy_generation
                WHERE fabric_group_id = %s
                """,
                (fabric_group_id,),
            )
            generation = int(cursor.fetchone()[0]) + 1
            snapshot, _ = build_policy_generation_snapshot(
                fabric_group_id, generation, data
            )
            digest = _policy_digest(snapshot)
            cursor.execute(
                f"""
                SELECT generation, activated_by, created_on
                FROM {self.config_schema}.llm_queue_policy_generation
                WHERE fabric_group_id = %s AND policy_digest = %s
                """,
                (fabric_group_id, digest),
            )
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(
                    f"""
                    INSERT INTO {self.config_schema}.llm_queue_policy_generation (
                        fabric_group_id,
                        generation,
                        policy_digest,
                        policy_snapshot,
                        activated_by
                    ) VALUES (%s, %s, %s, %s, %s)
                    RETURNING created_on
                    """,
                    (fabric_group_id, generation, digest, Jsonb(snapshot), actor_id),
                )
                created_on = cursor.fetchone()[0]
                activated_by = actor_id
            else:
                generation = int(existing[0])
                activated_by = str(existing[1])
                created_on = existing[2]
            cursor.execute(
                f"""
                UPDATE {self.config_schema}.llm_queue_fabric_config
                SET active_policy_generation = %s, updated_on = NOW()
                WHERE fabric_group_id = %s
                """,
                (generation, fabric_group_id),
            )
            conn.commit()
            return ActivatedPolicy(
                fabric_group_id=fabric_group_id,
                generation=generation,
                policy_digest=digest,
                activated_by=activated_by,
                created_on=created_on,
            )
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def activate_policy_revision(
        self,
        fabric_group_id: str,
        generation: int,
        actor_id: str,
        *,
        expected_generation: int | None = None,
    ) -> ActivatedPolicy:
        """Activate a validated saved snapshot without modifying draft settings."""
        if type(generation) is not int or not 1 <= generation <= 2**53 - 1:
            raise ValueError('routing_policy_invalid')
        if not _ACTOR_ID_RE.fullmatch(actor_id):
            raise ValueError('routing_policy_invalid')
        cursor = conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '3000ms'")
            cursor.execute(
                f"""SELECT active_policy_generation
                FROM {self.config_schema}.llm_queue_fabric_config
                WHERE fabric_group_id = %s FOR UPDATE""",
                (fabric_group_id,),
            )
            current = cursor.fetchone()
            if current is None:
                raise ValueError('routing_policy_revision_unavailable')
            if expected_generation is not None and current[0] != expected_generation:
                raise ValueError('routing_policy_activation_conflict')
            cursor.execute(
                f"""SELECT policy_digest, policy_snapshot, created_on
                FROM {self.config_schema}.llm_queue_policy_generation
                WHERE fabric_group_id = %s AND generation = %s""",
                (fabric_group_id, generation),
            )
            stored = cursor.fetchone()
            if stored is None:
                raise ValueError('routing_policy_revision_unavailable')
            digest, snapshot = str(stored[0]), stored[1]
            _admission_policy_from_snapshot(
                fabric_group_id, generation, digest, snapshot
            )
            cursor.execute(
                f"""UPDATE {self.config_schema}.llm_queue_fabric_config
                SET active_policy_generation = %s, updated_on = NOW()
                WHERE fabric_group_id = %s""",
                (generation, fabric_group_id),
            )
            conn.commit()
            return ActivatedPolicy(
                fabric_group_id=fabric_group_id,
                generation=generation,
                policy_digest=digest,
                activated_by=actor_id,
                created_on=stored[2],
            )
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def load_active_admission_policy(self, fabric_group_id: str) -> AdmissionPolicy:
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            cursor.execute(
                f"""
                SELECT active_policy_generation
                FROM {self.config_schema}.llm_queue_fabric_config
                WHERE fabric_group_id = %s AND enabled = true
                """,
                (fabric_group_id,),
            )
            active = cursor.fetchone()
            if active is None or active[0] is None:
                raise ValueError('LLM routing policy is not active')
            generation = int(active[0])
            cursor.execute(
                f"""
                SELECT policy_digest, policy_snapshot
                FROM {self.config_schema}.llm_queue_policy_generation
                WHERE fabric_group_id = %s AND generation = %s
                """,
                (fabric_group_id, generation),
            )
            stored = cursor.fetchone()
            if stored is None or not isinstance(stored[1], dict):
                raise ValueError('LLM routing policy generation is unavailable')
            digest, snapshot = str(stored[0]), stored[1]
            policy = _admission_policy_from_snapshot(
                fabric_group_id, generation, digest, snapshot
            )
            conn.commit()
            return policy
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def load_runtime_dispatch_policy(self, fabric_group_id: str) -> dict[str, Any]:
        """Load the immutable active dispatch policy used by the runtime."""
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            conn.policy_read_deadline = time.monotonic() + 5.0
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            cursor.execute(
                f"""
                SELECT config.active_policy_generation,
                       generation.policy_digest, generation.policy_snapshot
                FROM {self.config_schema}.llm_queue_fabric_config config
                LEFT JOIN {self.config_schema}.llm_queue_policy_generation generation
                  ON generation.fabric_group_id = config.fabric_group_id
                 AND generation.generation = config.active_policy_generation
                WHERE config.fabric_group_id = %s AND config.enabled = true
                """,
                (fabric_group_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError('LLM routing fabric is not configured')
            if row[0] is None:
                data = self._load_desired_policy(cursor, fabric_group_id, lock=False)
                policy = persisted_dispatch_policy(data)
                policy['policy_generation'] = None
                policy['policy_digest'] = None
                conn.commit()
                return policy
            if not isinstance(row[2], dict) or not isinstance(
                row[2].get('dispatch'), dict
            ):
                raise ValueError('LLM routing policy generation is unavailable')
            digest = str(row[1])
            if _policy_digest(row[2]) != digest:
                raise ValueError('LLM routing policy digest mismatch')
            policy = validate_dispatch_policy(row[2]['dispatch'])
            policy['policy_generation'] = int(row[0])
            policy['policy_digest'] = digest
            conn.commit()
            return policy
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            if conn is not None:
                conn.policy_read_deadline = None
            self._close_connection(conn)

    def load_routing_diagnostics(
        self, fabric_group_id: str, limit: int = 50
    ) -> dict[str, Any]:
        if not 1 <= limit <= 250:
            raise ValueError('Invalid routing diagnostic limit')
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            cursor.execute(
                f"""
                SELECT config.active_policy_generation, generation.policy_digest,
                       generation.policy_snapshot
                FROM {self.config_schema}.llm_queue_fabric_config config
                LEFT JOIN {self.config_schema}.llm_queue_policy_generation generation
                  ON generation.fabric_group_id = config.fabric_group_id
                 AND generation.generation = config.active_policy_generation
                WHERE config.fabric_group_id = %s
                """,
                (fabric_group_id,),
            )
            policy = cursor.fetchone()
            if policy is None:
                raise ValueError('LLM routing fabric is not configured')
            scheduler_config = None
            if policy[0] is not None:
                from marie.serve.runtimes.gateway.marie.scheduler_observation import (
                    scheduler_observation,
                )

                if not isinstance(policy[2], dict) or _policy_digest(policy[2]) != str(
                    policy[1]
                ):
                    raise ValueError('LLM routing policy digest mismatch')
                scheduler_config = scheduler_observation(
                    policy[2], int(policy[0]), limit=limit
                )
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE projection_state <> 'projected'),
                    COUNT(*) FILTER (WHERE routing_source = 'automatic'),
                    COUNT(*) FILTER (WHERE routing_source = 'operator-override')
                FROM {self.config_schema}.llm_job_route
                WHERE fabric_group_id = %s
                """,
                (fabric_group_id,),
            )
            routing = cursor.fetchone()
            cursor.execute(
                f"""
                SELECT pool.pool_id,
                       COUNT(route.work_unit_id) AS accepted,
                       COUNT(route.work_unit_id) FILTER (
                           WHERE job.state = 'completed'
                       ) AS completed,
                       COUNT(route.work_unit_id) FILTER (
                           WHERE route.projection_state <> 'projected'
                       ) AS projection_pending,
                       COUNT(route.work_unit_id) FILTER (
                           WHERE job.state NOT IN (
                               'completed', 'skipped', 'expired', 'cancelled', 'failed'
                           )
                       ) AS drain_references,
                       COUNT(*) OVER() AS pool_count
                FROM {self.config_schema}.llm_queue_pool pool
                LEFT JOIN {self.config_schema}.llm_job_route route
                  ON route.fabric_group_id = pool.fabric_group_id
                 AND route.pool_id = pool.pool_id
                LEFT JOIN {self.config_schema}.job job
                  ON job.id = route.work_unit_id
                WHERE pool.fabric_group_id = %s
                GROUP BY pool.pool_id, pool.sort_order
                ORDER BY pool.sort_order, pool.pool_id
                LIMIT %s
                """,
                (fabric_group_id, limit + 1),
            )
            rows = cursor.fetchall()
            cursor.execute(
                f"""
                SELECT job_id, work_unit_id, policy_generation, policy_digest,
                       rule_digest, effective_page_count, pool_id,
                       logical_endpoint_group_id, endpoint_revision,
                       estimator_version, routing_source, routing_actor,
                       routing_reason, projection_state
                FROM {self.config_schema}.llm_job_route
                WHERE fabric_group_id = %s
                ORDER BY created_on DESC, work_unit_id
                LIMIT %s
                """,
                (fabric_group_id, limit + 1),
            )
            route_rows = cursor.fetchall()
            conn.commit()
            visible = rows[:limit]
            visible_routes = route_rows[:limit]
            return {
                'scheduler_config': scheduler_config,
                'policy': {
                    'desired_generation': (
                        int(policy[0]) if policy[0] is not None else None
                    ),
                    'desired_digest': str(policy[1]) if policy[1] is not None else None,
                },
                'routing': {
                    'projection_pending_count': int(routing[0]),
                    'matched': {
                        'automatic': int(routing[1]),
                        'operator_override': int(routing[2]),
                    },
                    'rejected': {},
                },
                'pools': [
                    {
                        'pool_id': str(row[0]),
                        'accepted': int(row[1]),
                        'completed': int(row[2]),
                        'projection_pending': int(row[3]),
                        'drain_references': int(row[4]),
                    }
                    for row in visible
                ],
                'pool_count': int(rows[0][5]) if rows else 0,
                'pools_truncated': len(rows) > limit,
                'recent_routes': [
                    {
                        'job_id': str(row[0]),
                        'work_unit_id': str(row[1]),
                        'policy_generation': int(row[2]),
                        'policy_digest': str(row[3]),
                        'rule_digest': str(row[4]),
                        'effective_page_count': (
                            int(row[5]) if row[5] is not None else None
                        ),
                        'pool_id': str(row[6]),
                        'endpoint_group_id': str(row[7]),
                        'endpoint_revision': str(row[8]),
                        'estimator_version': str(row[9]),
                        'routing_source': str(row[10]),
                        'routing_actor': (
                            str(row[11]) if row[11] is not None else None
                        ),
                        'routing_reason': (
                            str(row[12]) if row[12] is not None else None
                        ),
                        'projection_state': str(row[13]),
                    }
                    for row in visible_routes
                ],
                'recent_routes_truncated': len(route_rows) > limit,
            }
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def routing_resource_references(
        self,
        *,
        fabric_group_id: str,
        resource_type: str,
        resource_id: str,
        revision: str | None,
        limit: int,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 250:
            raise ValueError('Invalid routing reference limit')
        where, params = _routing_reference_predicate(
            resource_type, resource_id, revision
        )
        cursor = None
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SET LOCAL statement_timeout = '1500ms'")
            base = f"""
                FROM {self.config_schema}.llm_job_route route
                LEFT JOIN {self.config_schema}.llm_queue_policy_generation policy
                  ON policy.fabric_group_id = route.fabric_group_id
                 AND policy.generation = route.policy_generation
                LEFT JOIN {self.config_schema}.job job
                  ON job.id = route.work_unit_id
                WHERE route.fabric_group_id = %s AND {where}
                  AND (
                    %s = 'policy_generation'
                    OR job.state NOT IN (
                        'completed', 'skipped', 'expired', 'cancelled', 'failed'
                    )
                  )
            """
            all_params = (fabric_group_id, *params, resource_type)
            cursor.execute(f'SELECT COUNT(*) {base}', all_params)
            count = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT route.work_unit_id {base}
                ORDER BY route.created_on DESC, route.work_unit_id
                LIMIT %s
                """,
                (*all_params, limit),
            )
            samples = [str(row[0]) for row in cursor.fetchall()]
            conn.commit()
            return {
                'postgres': count,
                'samples': samples,
                'truncated': count > len(samples),
            }
        except Exception:
            if conn is not None:
                conn.rollback()
            raise
        finally:
            self._close_cursor(cursor)
            self._close_connection(conn)

    def _load_desired_policy(
        self, cursor: Any, fabric_group_id: str, *, lock: bool
    ) -> dict[str, Any]:
        suffix = ' FOR UPDATE' if lock else ''
        cursor.execute(
            f"""
            SELECT policy, total_concurrent_dispatch, enabled, metadata,
                   active_policy_generation
            FROM {self.config_schema}.llm_queue_fabric_config
            WHERE fabric_group_id = %s{suffix}
            """,
            (fabric_group_id,),
        )
        fabric = cursor.fetchone()
        if fabric is None:
            raise ValueError('LLM routing fabric is not configured')
        cursor.execute(
            f"""
            SELECT pool_id, display_name, endpoint_url, quantum,
                   min_concurrent, max_concurrent, max_burst_per_visit,
                   enabled, metadata
            FROM {self.config_schema}.llm_queue_pool
            WHERE fabric_group_id = %s
            ORDER BY sort_order ASC, pool_id ASC
            LIMIT {MAX_DATABASE_LANES + 1}
            """,
            (fabric_group_id,),
        )
        rows = cursor.fetchall()
        if len(rows) > MAX_DATABASE_LANES:
            raise ValueError('LLM queue database policy exceeds 1000 lanes')
        return {
            'policy': fabric[0],
            'total_concurrent_dispatch': fabric[1],
            'enabled': fabric[2],
            'metadata': fabric[3],
            'active_policy_generation': fabric[4],
            'lanes': [
                {
                    'pool_id': row[0],
                    'display_name': row[1],
                    'endpoint_url': row[2],
                    'quantum': row[3],
                    'min_concurrent': row[4],
                    'max_concurrent': row[5],
                    'max_burst_per_visit': row[6],
                    'enabled': row[7],
                    'metadata': row[8],
                }
                for row in rows
            ],
        }


def _admission_policy_from_snapshot(
    fabric_group_id: str, generation: int, digest: str, snapshot: dict[str, Any]
) -> AdmissionPolicy:
    if (
        not isinstance(snapshot, dict)
        or snapshot.get('fabric_group_id') != fabric_group_id
    ):
        raise ValueError('routing_policy_invalid')
    if _policy_digest(snapshot) != digest:
        raise ValueError('LLM routing policy digest mismatch')
    validate_dispatch_policy(snapshot.get('dispatch', {}))
    admission_snapshot = snapshot.get('admission')
    if not isinstance(admission_snapshot, dict):
        raise ValueError('LLM admission policy snapshot is unavailable')
    dispatch_lanes = {
        lane['pool_id']: lane
        for lane in snapshot.get('dispatch', {}).get('lanes', [])
        if isinstance(lane, dict) and isinstance(lane.get('pool_id'), str)
    }
    rows = [
        {
            'pool_id': rule['pool_id'],
            'enabled': rule['enabled'],
            'metadata': {
                'admission': rule['admission'],
                'llm_dispatch': {
                    'schema_version': 1,
                    'endpoint_id': dispatch_lanes[rule['pool_id']].get(
                        'endpoint_group_id'
                    )
                    or dispatch_lanes[rule['pool_id']]['endpoint_id'],
                    'revision': dispatch_lanes[rule['pool_id']]['revision'],
                },
            },
        }
        for rule in admission_snapshot.get('rules', [])
    ]
    policy = AdmissionPolicy.from_rows(fabric_group_id, generation, rows)
    return replace(policy, policy_digest=digest)


def _sql_identifier(value: Any, *, label: str) -> str:
    identifier = str(value).strip()
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid LLM queue scheduler {label}: {value!r}")
    return identifier


def _routing_reference_predicate(
    resource_type: str, resource_id: str, revision: str | None
) -> tuple[str, tuple[Any, ...]]:
    if resource_type == 'pool':
        return 'route.pool_id = %s', (resource_id,)
    if resource_type == 'policy_generation':
        try:
            generation = int(resource_id)
        except ValueError:
            raise ValueError('Invalid routing policy generation') from None
        if not 1 <= generation <= 2**53 - 1:
            raise ValueError('Invalid routing policy generation')
        return 'route.policy_generation = %s', (generation,)
    if resource_type == 'endpoint_group':
        if revision is None:
            return 'route.logical_endpoint_group_id = %s', (resource_id,)
        return (
            'route.logical_endpoint_group_id = %s AND route.endpoint_revision = %s',
            (resource_id, revision),
        )
    if resource_type == 'replica':
        revision_filter = (
            "AND endpoint_group->>'revision' = %s" if revision is not None else ''
        )
        params: tuple[Any, ...] = (
            (resource_id, revision) if revision is not None else (resource_id,)
        )
        legacy = 'FALSE'
        if revision in {None, 'r1'}:
            legacy = """(
                route.logical_endpoint_group_id = %s
                AND route.endpoint_revision = 'r1'
                AND EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(
                        policy.policy_snapshot->'dispatch'->'endpoints'
                    ) endpoint
                    WHERE endpoint->>'endpoint_id' = %s
                )
            )"""
            params = (*params, resource_id, resource_id)
        return (
            f"""(
                EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(
                        policy.policy_snapshot->'dispatch'->'endpoint_groups'
                    ) endpoint_group
                    CROSS JOIN LATERAL jsonb_array_elements(
                        endpoint_group->'replicas'
                    ) replica
                    WHERE replica->>'replica_id' = %s
                      AND endpoint_group->>'group_id' = route.logical_endpoint_group_id
                      AND endpoint_group->>'revision' = route.endpoint_revision
                      {revision_filter}
                ) OR {legacy}
            )""",
            params,
        )
    raise ValueError('Invalid routing resource type')


def _policy_digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot,
        separators=(',', ':'),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
