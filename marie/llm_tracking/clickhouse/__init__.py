"""
ClickHouse Module - Analytics storage for LLM tracking.

Components:
- ClickHouseClientManager: Singleton client for ClickHouse connections
- ClickHouseWriter: Batched writer for high-throughput inserts

Note: ClickHouse schemas are in config/clickhouse/schema/llm_tracking.sql
"""

from marie.llm_tracking.clickhouse.client import ClickHouseClientManager
from marie.llm_tracking.clickhouse.writer import ClickHouseWriter, TableName

__all__ = ["ClickHouseClientManager", "ClickHouseWriter", "TableName"]
