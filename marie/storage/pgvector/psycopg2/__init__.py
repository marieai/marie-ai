"""Compatibility pgvector registration using psycopg 3."""

import logging

import numpy as np
import psycopg
from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format
from psycopg.types import TypeInfo

from ..utils import from_db, to_db

__all__ = ["register_vector"]

logger = logging.getLogger(__name__)


class VectorDumper(Dumper):
    format = Format.TEXT

    def dump(self, vector):
        return to_db(vector).encode("utf-8")


class VectorLoader(Loader):
    format = Format.TEXT

    def load(self, value):
        if isinstance(value, memoryview):
            value = value.tobytes()
        return from_db(value.decode("utf-8"))


def register_vector(conn_or_curs=None, raise_on_missing: bool = False):
    """Register the vector type with a psycopg 3 connection or cursor."""
    context = getattr(conn_or_curs, "connection", conn_or_curs)
    info = TypeInfo.fetch(context, "vector")
    if info is None:
        if raise_on_missing:
            raise psycopg.ProgrammingError("vector type not found in the database")
        logger.debug("pgvector extension is not installed; registration skipped")
        return False

    info.register(context)
    dumper = type("VectorDumper", (VectorDumper,), {"oid": info.oid})
    context.adapters.register_dumper(np.ndarray, dumper)
    context.adapters.register_loader(info.oid, VectorLoader)
    return True
