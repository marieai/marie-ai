from functools import cached_property
from typing import Optional, TypeVar, Dict

from docarray import DocumentArray

from marie.excepts import BadRequestType
from marie.helper import typename, random_identity
from marie.types.mixin import ProtoTypeMixin
from marie.types.request import Request

RequestSourceType = TypeVar("RequestSourceType", str, Dict, bytes)


class Header:
    request_id: str
    exec_endpoint: str
    target_executor: str


class DataRequestProto:
    def __init__(self):
        self.docs: "DocumentArray"
        self.docs_bytes: bytes
        self.header: Header = Header()
        self.parameters = {}


class DataRequest(Request):
    """Represents a DataRequest used for exchanging DocumentArrays to and within a Flow"""

    class _DataContent:
        def __init__(self, content: "DataRequestProto"):
            self._content = content
            self._loaded_doc_array = None

        #
        # def __len__(self):
        #     return len(self._loaded_doc_array)

        @property
        def docs(self) -> "DocumentArray":
            """Get the :class: `DocumentArray` with sequence `data.docs` as content.

            .. # noqa: DAR201"""
            if not self._loaded_doc_array:
                self._loaded_doc_array = DocumentArray.from_bytes(self._content.docs_bytes)

            return self._loaded_doc_array

        @docs.setter
        def docs(self, value: DocumentArray):
            """Override the DocumentArray with the provided one

            :param value: a DocumentArray
            """
            self.set_docs_convert_arrays(value, None)

        def set_docs_convert_arrays(self, value: DocumentArray, ndarray_type: Optional[str] = None):
            """ " Convert embedding and tensor to given type, then set DocumentArray

            :param value: a DocumentArray
            :param ndarray_type: type embedding and tensor will be converted to
            """
            if value is not None:
                self._loaded_doc_array = None
                self._content.docs = value

        @property
        def docs_bytes(self) -> bytes:
            """Get the :class: `DocumentArray` with sequence `data.docs` as content.

            .. # noqa: DAR201"""
            return self._content.docs_bytes

        @docs_bytes.setter
        def docs_bytes(self, value: bytes):
            """Override the DocumentArray with the provided one

            :param value: a DocumentArray
            """
            if value:
                self._loaded_doc_array = None
                self._content.docs_bytes = value

    def __init__(
        self,
        request: Optional[RequestSourceType] = None,
    ):

        self.buffer = None

        try:
            if isinstance(request, DataRequestProto):
                self._pb_body = request
            elif isinstance(request, dict):
                self._pb_body = DataRequestProto()
                # json_format.ParseDict(request, self._pb_body)
            elif isinstance(request, str):
                self._pb_body = DataRequestProto()
                # json_format.Parse(request, self._pb_body)
            elif isinstance(request, bytes):
                self.buffer = request
            elif request is not None:
                # note ``None`` is not considered as a bad type
                raise ValueError(f"{typename(request)} is not recognizable")
            else:
                self._pb_body = DataRequestProto()
                self._pb_body.header.request_id = random_identity()

        except Exception as ex:
            raise BadRequestType(f"fail to construct a {self.__class__} object from {request}") from ex

    @property
    def is_decompressed(self) -> bool:
        """
        Checks if the underlying proto object was already deserialized

        :return: True if the proto was deserialized before
        """
        return self.buffer is None

    @property
    def proto(self) -> "DataRequestProto":
        """
        Cast ``self`` to a :class:`jina_pb2.DataRequestProto`. Laziness will be broken and serialization will be recomputed when calling
        :meth:`SerializeToString`.
        :return: protobuf instance
        """
        if not self.is_decompressed:
            self._decompress()
        return self._pb_body

    def _decompress(self):
        # raise NotImplemented
        self._pb_body = DataRequestProto()
        # self._pb_body.ParseFromString(self.buffer)
        self.buffer = None

    @property
    def docs(self) -> "DocumentArray":
        """Get the :class: `DocumentArray` with sequence `data.docs` as content.

        .. # noqa: DAR201"""
        return self.data.docs

    @cached_property
    def data(self) -> "DataRequest._DataContent":
        """Get the data contaned in this data request

        :return: the data content as an instance of _DataContent wrapping docs
        """
        return DataRequest._DataContent(self.proto)
        # return DataRequest._DataContent(self.proto.data)


class Response(DataRequest):
    """
    Response is the :class:`Request` object returns from the flow. Right now it shares the same representation as
    :class:`Request`.
    """
