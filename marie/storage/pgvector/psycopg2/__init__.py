import logging
import numpy as np
import psycopg2
from psycopg2.extensions import adapt, new_type, register_adapter, register_type
from ..utils import from_db, to_db

__all__ = ['register_vector']

logger = logging.getLogger(__name__)


class VectorAdapter(object):
    def __init__(self, vector):
        self._vector = vector

    def getquoted(self):
        return adapt(to_db(self._vector)).getquoted()


def cast_vector(value, cur):
    return from_db(value)


def register_vector(conn_or_curs=None, raise_on_missing: bool = False):
    """
    Register the vector type with psycopg2.

    :param conn_or_curs: A psycopg2 connection or cursor
    :param raise_on_missing: If True, raise an error when the vector type is not found.
                             If False (default), log a warning and return False.
    :return: True if registration succeeded, False if vector type not found
    """
    cur = conn_or_curs.cursor() if hasattr(conn_or_curs, 'cursor') else conn_or_curs

    try:
        cur.execute('SELECT NULL::vector')
        oid = cur.description[0][1]
    except psycopg2.errors.UndefinedObject:
        if raise_on_missing:
            raise psycopg2.ProgrammingError('vector type not found in the database')
        logger.debug(
            'pgvector extension not installed in database; vector type registration skipped'
        )
        # Rollback the failed transaction to keep the connection usable
        if hasattr(conn_or_curs, 'rollback'):
            conn_or_curs.rollback()
        return False

    vector = new_type((oid,), 'VECTOR', cast_vector)
    register_type(vector)
    register_adapter(np.ndarray, VectorAdapter)
    return True
