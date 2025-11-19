"""
Base HITL Executor with database interaction logic.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import pool

from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger


class BaseHitlExecutor(MarieExecutor):
    """
    Base executor for HITL operations with database interaction.

    Provides common functionality for:
    - Creating HITL requests in the database
    - Polling for human responses
    - Handling timeouts
    - Managing request lifecycle
    """

    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        poll_interval: int = 5,
        max_poll_attempts: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize base HITL executor.

        Args:
            db_config: Database configuration dict with keys:
                - host: Database host (default: from env or localhost)
                - port: Database port (default: from env or 5432)
                - database: Database name (default: from env or marie_studio)
                - user: Database user (default: from env or postgres)
                - password: Database password (default: from env)
            poll_interval: Seconds between polling for responses (default: 5)
            max_poll_attempts: Maximum polling attempts before timeout (default: None = unlimited)
            **kwargs: Additional executor arguments
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__)

        # Database configuration from environment or defaults
        self.db_config = db_config or self._get_default_db_config()
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

        # Database connection pool
        self.db_pool = None
        self._init_db_pool()

        self.logger.info(f"HITL Executor initialized: {self.__class__.__name__}")
        self.logger.info(
            f"Database: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        self.logger.info(f"Poll interval: {self.poll_interval}s")

    def _get_default_db_config(self) -> Dict[str, Any]:
        """Get default database configuration from environment variables."""
        # Parse DATABASE_URL if available
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Parse postgres://user:password@host:port/database
            import re

            match = re.match(
                r"postgres(?:ql)?://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", database_url
            )
            if match:
                user, password, host, port, database = match.groups()
                return {
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "user": user,
                    "password": password,
                }

        # Fallback to individual env vars or defaults
        return {
            "host": os.getenv("MARIE_STUDIO_DB_HOST", "localhost"),
            "port": int(os.getenv("MARIE_STUDIO_DB_PORT", "5432")),
            "database": os.getenv("MARIE_STUDIO_DB_NAME", "marie_studio"),
            "user": os.getenv("MARIE_STUDIO_DB_USER", "postgres"),
            "password": os.getenv("MARIE_STUDIO_DB_PASSWORD", ""),
        }

    def _init_db_pool(self):
        """Initialize database connection pool."""
        try:
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise

    def _get_db_connection(self):
        """Get a database connection from the pool."""
        return self.db_pool.getconn()

    def _release_db_connection(self, conn):
        """Release a database connection back to the pool."""
        self.db_pool.putconn(conn)

    async def create_hitl_request(
        self,
        dag_id: str,
        job_id: str,
        request_type: str,
        title: str,
        description: Optional[str],
        priority: str,
        context_data: Dict[str, Any],
        config: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """
        Create a HITL request in the database.

        Args:
            dag_id: DAG ID
            job_id: Job ID
            request_type: Type of request (approval, correction, router)
            title: Request title
            description: Request description
            priority: Priority level (low, medium, high, critical)
            context_data: Context data for the request
            config: Request configuration (approval_type, fields, thresholds, etc.)
            timeout_seconds: Timeout in seconds (optional)

        Returns:
            str: Request ID (UUID)
        """
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Calculate timeout timestamp if provided
            timeout_at = None
            if timeout_seconds:
                timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

            # Extract assigned users/roles from config
            assigned_to = config.get("assigned_to", {})
            assigned_user_ids = assigned_to.get("user_ids", [])
            assigned_roles = assigned_to.get("roles", [])

            # Insert HITL request
            query = """
                INSERT INTO marie_scheduler.hitl_requests (
                    dag_id, job_id, request_type, title, description, priority,
                    context_data, config, timeout_at, assigned_user_ids, assigned_roles,
                    status, require_all_approval
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s
                )
                RETURNING id
            """

            require_all = config.get("require_all_approval", False)

            cursor.execute(
                query,
                (
                    dag_id,
                    job_id,
                    request_type,
                    title,
                    description,
                    priority,
                    json.dumps(context_data),
                    json.dumps(config),
                    timeout_at,
                    assigned_user_ids,
                    assigned_roles,
                    "pending",
                    require_all,
                ),
            )

            result = cursor.fetchone()
            request_id = result["id"]

            conn.commit()
            cursor.close()

            self.logger.info(
                f"Created HITL request: {request_id} (type: {request_type}, priority: {priority})"
            )
            return request_id

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to create HITL request: {e}")
            raise
        finally:
            self._release_db_connection(conn)

    async def poll_for_response(
        self,
        request_id: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Poll for human response to a HITL request.

        Args:
            request_id: Request ID to poll
            timeout_seconds: Maximum time to wait (optional)

        Returns:
            Dict with response data including:
                - status: Request status (completed, timeout, cancelled)
                - decision: Approval decision (if approval type)
                - corrected_data: Corrected data (if correction type)
                - feedback: User feedback
                - responded_by: User ID who responded

        Raises:
            TimeoutError: If timeout is reached before response
        """
        start_time = datetime.utcnow()
        poll_count = 0

        while True:
            # Check max attempts
            if self.max_poll_attempts and poll_count >= self.max_poll_attempts:
                raise TimeoutError(
                    f"Max poll attempts ({self.max_poll_attempts}) reached for request {request_id}"
                )

            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    # Handle timeout
                    return await self._handle_timeout(request_id)

            # Poll database for response
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Get request status and latest response
                query = """
                    SELECT
                        r.id,
                        r.status,
                        r.config,
                        r.require_all_approval,
                        resp.decision,
                        resp.corrected_data,
                        resp.feedback,
                        resp.responded_by,
                        resp.created_at as response_time
                    FROM marie_scheduler.hitl_requests r
                    LEFT JOIN marie_scheduler.hitl_responses resp ON r.id = resp.request_id
                    WHERE r.id = %s
                    ORDER BY resp.created_at DESC
                    LIMIT 1
                """

                cursor.execute(query, (request_id,))
                result = cursor.fetchone()
                cursor.close()

                if not result:
                    raise ValueError(f"HITL request {request_id} not found")

                status = result["status"]

                # Check if request is completed
                if status == "completed":
                    self.logger.info(f"HITL request {request_id} completed")
                    return {
                        "status": "completed",
                        "decision": result.get("decision"),
                        "corrected_data": result.get("corrected_data"),
                        "feedback": result.get("feedback"),
                        "responded_by": result.get("responded_by"),
                        "response_time": result.get("response_time"),
                    }

                # Check if request is cancelled
                if status == "cancelled":
                    self.logger.warning(f"HITL request {request_id} was cancelled")
                    return {
                        "status": "cancelled",
                        "decision": None,
                        "corrected_data": None,
                        "feedback": "Request was cancelled",
                    }

                # Check if request is timeout
                if status == "timeout":
                    self.logger.warning(f"HITL request {request_id} timed out")
                    return await self._handle_timeout(request_id)

            finally:
                self._release_db_connection(conn)

            # Wait before next poll
            poll_count += 1
            await asyncio.sleep(self.poll_interval)

    async def _handle_timeout(self, request_id: str) -> Dict[str, Any]:
        """
        Handle HITL request timeout.

        Args:
            request_id: Request ID that timed out

        Returns:
            Dict with timeout handling result
        """
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Get request config to determine timeout strategy
            cursor.execute(
                "SELECT config FROM marie_scheduler.hitl_requests WHERE id = %s",
                (request_id,),
            )
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"Request {request_id} not found")

            config = result["config"]
            timeout_config = config.get("timeout", {})
            strategy = timeout_config.get("strategy", "use_default")

            self.logger.info(
                f"Handling timeout for request {request_id} with strategy: {strategy}"
            )

            # Update request status to timeout
            cursor.execute(
                """
                UPDATE marie_scheduler.hitl_requests
                SET status = 'timeout'
                WHERE id = %s
                """,
                (request_id,),
            )
            conn.commit()
            cursor.close()

            # Return based on strategy
            if strategy == "use_default":
                default_option = config.get("default_option")
                return {
                    "status": "timeout",
                    "decision": default_option,
                    "corrected_data": None,
                    "feedback": "Request timed out - using default option",
                }
            elif strategy == "use_original":
                # For correction type - use original data
                return {
                    "status": "timeout",
                    "decision": None,
                    "corrected_data": config.get("context_data", {}).get(
                        "original_data"
                    ),
                    "feedback": "Request timed out - using original data",
                }
            elif strategy == "fail":
                raise TimeoutError(
                    f"HITL request {request_id} timed out (strategy: fail)"
                )
            elif strategy == "skip_downstream":
                return {
                    "status": "timeout",
                    "decision": "skip",
                    "corrected_data": None,
                    "feedback": "Request timed out - skipping downstream tasks",
                }
            else:
                # Default behavior
                return {
                    "status": "timeout",
                    "decision": None,
                    "corrected_data": None,
                    "feedback": f"Request timed out (unknown strategy: {strategy})",
                }

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to handle timeout for request {request_id}: {e}")
            raise
        finally:
            self._release_db_connection(conn)

    def close(self):
        """Close database connections."""
        if self.db_pool:
            self.db_pool.closeall()
            self.logger.info("Database connection pool closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
