import abc
import json

__all__ = ['Address', 'PlainAddress', 'JsonAddress']


def b2str(i_b):
    return i_b.decode()


class Address(abc.ABC):
    """gRPC service address."""

    @abc.abstractmethod
    def __init__(self, addr, metadata=None):
        raise NotImplementedError

    @abc.abstractmethod
    def add_value(self):
        raise NotImplementedError

    @abc.abstractmethod
    def delete_value(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_value(cls, val, deserializer=None):
        raise NotImplementedError


class PlainAddress(Address):
    """Plain text address."""

    def __init__(self, addr, metadata=None):
        self._addr = addr

    def add_value(self):
        return self._addr

    def delete_value(self):
        return self._addr

    @classmethod
    def from_value(cls, val, deserializer=None):
        return b2str(val)


class JsonAddress(Address):
    """Json address."""

    add_op = 0
    delete_op = 1

    def __init__(self, addr, metadata=None, serializer=json.dumps):
        self._addr = addr
        self._metadata = metadata or {}
        self._serializer = serializer

    def add_value(self):
        return self._serializer(
            {
                'Op': self.add_op,
                'Addr': self._addr,
                'Metadata': self._serializer(self._metadata),
            }
        )

    def delete_value(self):
        return self._serializer(
            {
                'Op': self.delete_op,
                'Addr': self._addr,
                'Metadata': self._serializer(self._metadata),
            }
        )

    @classmethod
    def from_value(cls, val, deserializer=json.loads):
        addr_val = deserializer(b2str(val))
        addr_val['Metadata'] = deserializer(addr_val['Metadata'])
        addr_op = addr_val['Op']
        if addr_op == cls.add_op:
            return True, addr_val['Addr']
        elif addr_op == cls.delete_op:
            return False, addr_val['Addr']

        raise ValueError('invalid address value.')
