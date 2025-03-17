"""Base schema for data structures."""

from __future__ import annotations

import base64
import json
import logging
import pickle
import textwrap
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import filetype
from docarray import BaseDoc
from typing_extensions import Self

from marie._core.bridge.pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    JsonSchemaValue,
    PlainSerializer,
    SerializeAsAny,
    model_serializer,
    model_validator,
)

DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"

ImageType = Union[str, BytesIO]


class ObjectType(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()
    DOCUMENT = auto()
    MULTIMODAL = auto()


class Modality(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"


class BaseComponent(BaseDoc):
    """Base component object to capture class names."""

    @classmethod
    def class_name(cls) -> str:
        """
        Get the class name, used as a unique ID in serialization.

        This provides a key that makes serialization robust against actual class
        name changes.
        """
        return "base_component"

    def json(self, **kwargs: Any) -> str:
        return self.to_json(**kwargs)

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> Dict[str, Any]:
        data = handler(self)
        data["class_name"] = self.class_name()
        return data

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model_dump(**kwargs)

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()

        # remove attributes that are not pickleable -- kind of dangerous
        keys_to_remove = []
        for key, val in state["__dict__"].items():
            try:
                pickle.dumps(val)
            except Exception:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            logging.warning(f"Removing unpickleable attribute {key}")
            del state["__dict__"][key]

        # remove private attributes if they aren't pickleable -- kind of dangerous
        keys_to_remove = []
        private_attrs = state.get("__pydantic_private__", None)
        if private_attrs:
            for key, val in state["__pydantic_private__"].items():
                try:
                    pickle.dumps(val)
                except Exception:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                logging.warning(f"Removing unpickleable private attribute {key}")
                del state["__pydantic_private__"][key]

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Use the __dict__ and __init__ method to set state
        # so that all variables initialize
        try:
            self.__init__(**state["__dict__"])  # type: ignore
        except Exception:
            # Fall back to the default __setstate__ method
            # This may not work if the class had unpickleable attributes
            super().__setstate__(state)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        # In SimpleKVStore we rely on shallow coping. Hence, the data will be modified in the store directly.
        # And it is the same when the user is passing a dictionary to create a component. We can't modify the passed down dictionary.
        data = dict(data)
        if isinstance(kwargs, dict):
            data.update(kwargs)
        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class BaseNode(BaseComponent):
    """Base node Object.

    Generic abstract interface for retrievable nodes

    """

    embedding: Optional[List[float]] = Field(
        default=None, description="Embedding of the node."
    )

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering

    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )
    excluded_embed_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_separator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
        alias="metadata_seperator",
    )

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @abstractmethod
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        """Get object content."""

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_separator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    @abstractmethod
    def set_content(self, value: Any) -> None:
        """Set the content of the node."""


EmbeddingKind = Literal["sparse", "dense"]


class MediaResource(BaseDoc):
    """A container class for media content.

    This class represents a generic media resource that can be stored and accessed
    in multiple ways - as raw bytes, on the filesystem, or via URL. It also supports
    storing vector embeddings for the media content.

    Attributes:
        embeddings: Multi-vector dict representation of this resource for embedding-based search/retrieval
        text: Plain text representation of this resource
        data: Raw binary data of the media content
        mimetype: The MIME type indicating the format/type of the media content
        path: Local filesystem path where the media content can be accessed
        url: URL where the media content can be accessed remotely
    """

    embeddings: dict[EmbeddingKind, list[float]] | None = Field(
        default=None, description="Vector representation of this resource."
    )
    data: bytes | None = Field(
        default=None,
        exclude=True,
        description="base64 binary representation of this resource.",
    )
    text: str | None = Field(
        default=None, description="Text representation of this resource."
    )
    mimetype: str | None = Field(
        default=None, description="MIME type of this resource."
    )
    path: Path | None = Field(
        default=None, description="Filesystem path of this resource."
    )
    url: AnyUrl | None = Field(default=None, description="URL to reach this resource.")

    @model_validator(mode="after")
    def data_to_base64(self) -> Self:
        """If binary data was passed, store the resource as base64 and guess the mimetype when possible.

        In case the model was built passing binary data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.data:
            return self

        try:
            # Check if data is already base64 encoded
            decoded_data = base64.b64decode(self.data)
        except Exception:
            decoded_data = self.data
            # Not base64 - encode it
            self.data = base64.b64encode(self.data)

        if not self.mimetype:
            guess = filetype.guess(decoded_data)
            self.mimetype = guess.mime if guess else None

        return self

    @property
    def hash(self) -> str:
        """Generate a hash to uniquely identify the media resource.

        The hash is generated based on the available content (data, path, text or url).
        Returns an empty string if no content is available.
        """
        bits: list[str] = []
        if self.text is not None:
            bits.append(self.text)
        if self.data is not None:
            # Hash the binary data if available
            bits.append(str(sha256(self.data).hexdigest()))
        if self.path is not None:
            # Hash the file path if provided
            bits.append(str(sha256(str(self.path).encode("utf-8")).hexdigest()))
        if self.url is not None:
            # Use the URL string as basis for hash
            bits.append(str(sha256(str(self.url).encode("utf-8")).hexdigest()))

        doc_identity = "".join(bits)
        if not doc_identity:
            return ""
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())


class TextNode(BaseNode):
    """Provided for backward compatibility.

    Note: we keep the field with the typo "seperator" to maintain backward compatibility for
    serialized objects.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Make TextNode forward-compatible with Node by supporting 'text_resource' in the constructor."""
        if "text_resource" in kwargs:
            tr = kwargs.pop("text_resource")
            if isinstance(tr, MediaResource):
                kwargs["text"] = tr.text
            else:
                kwargs["text"] = tr["text"]
        super().__init__(*args, **kwargs)

    text: str = Field(default="", description="Text content of the node.")
    mimetype: str = Field(
        default="text/plain", description="MIME type of the node content."
    )
    start_char_idx: Optional[int] = Field(
        default=None, description="Start char index of the node."
    )
    end_char_idx: Optional[int] = Field(
        default=None, description="End char index of the node."
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and "
            "{metadata_str} placeholders."
        ),
    )

    @classmethod
    def class_name(cls) -> str:
        return "TextNode"

    @property
    def hash(self) -> str:
        doc_identity = str(self.text) + str(self.metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        return self.text_template.format(
            content=self.text, metadata_str=metadata_str
        ).strip()


class ImageNode(TextNode):
    """Node with image."""

    # TODO: store reference instead of actual image
    # base64 encoded image str
    image: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_mimetype: Optional[str] = None
    text_embedding: Optional[List[float]] = Field(
        default=None,
        description="Text embedding of image node, if text field is filled out",
    )

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.IMAGE

    @classmethod
    def class_name(cls) -> str:
        return "ImageNode"

    def resolve_image(self) -> ImageType:
        """Resolve an image such that PIL can read it."""
        if self.image is not None:
            import base64

            return BytesIO(base64.b64decode(self.image))
        elif self.image_path is not None:
            return self.image_path
        elif self.image_url is not None:
            # load image from URL
            import requests

            response = requests.get(self.image_url)
            return BytesIO(response.content)
        else:
            raise ValueError("No image found in node.")

    @property
    def hash(self) -> str:
        """Get hash of node."""
        # doc identity depends on if image, image_path, or image_url is set
        image_str = self.image or "None"
        image_path_str = self.image_path or "None"
        image_url_str = self.image_url or "None"
        image_text = self.text or "None"
        doc_identity = f"{image_str}-{image_path_str}-{image_url_str}-{image_text}"
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())
