from typing import Dict


class ProtoTypeMixin:
    """The base mixin class of all Marie types."""

    def to_json(self) -> str:
        raise NotImplemented

    def to_dict(self) -> Dict:
        """Return the object in Python dictionary.

        .. note::
            Array like object such as :class:`numpy.ndarray` (i.e. anything described as :class:`jina_pb2.NdArrayProto`)
            will be converted to Python list.

        :return: dict representation of the object
        """
        raise NotImplemented

    def to_bytes(self) -> bytes:
        """Return the serialized the message to a string.

        For more Pythonic code, please use ``bytes(...)``.

        :return: binary string representation of the object
        """
        raise NotImplemented

    def __getstate__(self):
        raise NotImplemented

    def __setstate__(self, state):
        self.__init__()
        raise NotImplemented

    @property
    def nbytes(self) -> int:
        """Return total bytes consumed by protobuf.

        :return: number of bytes
        """
        return len(bytes(self))

    def __getattr__(self, name: str):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented

    def clear(self) -> None:
        """Remove all values from all fields of this Document."""
        raise NotImplemented

    def pop(self, *fields) -> None:
        """Remove the values from the given fields of this Document.

        :param fields: field names
        """
        raise NotImplemented

    def __eq__(self, other):
        raise NotImplemented

    def __bytes__(self):
        return self.to_bytes()
