"""Marie AI persistence adapter for reusable LLM scheduler configuration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional

from marie.engine.llm_queue.admission_policy import AdmissionPolicy
from psycopg.types.json import Jsonb

from marie.logging_core.logger import MarieLogger
from marie.serve.runtimes.gateway.marie.dispatch_policy import (
    build_policy_generation_snapshot,
    validate_dispatch_policy,
)
from marie.storage.database.postgres import PostgresqlMixin

DEFAULT_SCHEDULER_CONFIG_SCHEMA = "marie_scheduler"
DEFAULT_FABRIC_CONFIG_TABLE = "llm_queue_fabric_config"
DEFAULT_POOL_TABLE = "llm_queue_pool"
MAX_DATABASE_LANES = 1000
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ACTOR_ID_RE = re.compile(r'^[A-Za-z0-9_.@-]{1,128}$')


@dataclass(frozen=True, slots=True)
class ActivatedPolicy:
    fabric_group_id: str
    generation: int
    policy_digest: str
    activated_by: str
    created_on: datetime


class PostgresSchedulerConfigRepository(PostgresqlMixin):
    """Load engine scheduler configuration from Marie's PostgreSQL store."""

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
            if _policy_digest(snapshot) != digest:
                raise ValueError('LLM routing policy digest mismatch')
            validate_dispatch_policy(snapshot.get('dispatch', {}))
            admission_snapshot = snapshot.get('admission')
            if not isinstance(admission_snapshot, dict):
                raise ValueError('LLM admission policy snapshot is unavailable')
            rows = [
                {
                    'pool_id': rule['pool_id'],
                    'enabled': rule['enabled'],
                    'metadata': {'admission': rule['admission']},
                }
                for rule in admission_snapshot.get('rules', [])
            ]
            policy = AdmissionPolicy.from_rows(fabric_group_id, generation, rows)
            conn.commit()
            return replace(policy, policy_digest=digest)
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
                   admission_mode, active_policy_generation
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
            'admission_mode': fabric[4],
            'active_policy_generation': fabric[5],
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


def _sql_identifier(value: Any, *, label: str) -> str:
    identifier = str(value).strip()
    if not _SQL_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid LLM queue scheduler {label}: {value!r}")
    return identifier


def _policy_digest(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot,
        separators=(',', ':'),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
