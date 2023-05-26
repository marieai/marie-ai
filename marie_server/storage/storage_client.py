import abc
from typing import Any, Optional, List, Dict


class StorageArea(abc.ABC):
    """
    Abstract base class for a KV storage clients.
    """

    @abc.abstractmethod
    async def internal_kv_get(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> Optional[Any]:
        """
        Retrieves a value from the internal key-value store.
        :param key: the key to retrieve
        :param namespace: the namespace to retrieve the key from
        :param timeout: the timeout in seconds
        :return: the value or None if the key does not exist
        """

    @abc.abstractmethod
    async def internal_kv_multi_get(
        self,
        keys: List[bytes],
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> Dict[bytes, bytes]:
        """
        Retrieves multiple values from the internal key-value store.
        :param keys: the keys to retrieve
        :param namespace: the namespace to retrieve the key from
        :param timeout: the timeout in seconds
        :return: a dictionary of key-value pairs
        """

    @abc.abstractmethod
    async def internal_kv_put(
        self,
        key: bytes,
        value: bytes,
        overwrite: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        """
        Stores a value in the internal key-value store.
        :param key: the key to store
        :param value: the value to store
        :param overwrite: whether to overwrite an existing value
        :param namespace: the namespace to store the key in
        :param timeout: the timeout in seconds
        :return: the version of the key-value pair
        """

    @abc.abstractmethod
    async def internal_kv_del(
        self,
        key: bytes,
        del_by_prefix: bool,
        namespace: Optional[bytes],
        timeout: Optional[float] = None,
    ) -> int:
        """
        Deletes a value from the internal key-value store.
        :param key: the key to delete
        :param del_by_prefix: whether to delete by prefix
        :param namespace: the namespace to delete the key from
        :param timeout: the timeout in seconds
        :return: the version of the key-value pair
        """

    @abc.abstractmethod
    async def internal_kv_exists(
        self, key: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> bool:
        """
        Checks if a key exists in the internal key-value store.
        :param key: the key to check
        :param namespace: the namespace to check the key from
        :param timeout: the timeout in seconds
        :return: True if the key exists, False otherwise
        """

    @abc.abstractmethod
    async def internal_kv_keys(
        self, prefix: bytes, namespace: Optional[bytes], timeout: Optional[float] = None
    ) -> List[bytes | str]:
        """
        Returns a list of keys in the internal key-value store.
        :param prefix: the prefix to filter on
        :param namespace: the namespace to retrieve the key from
        :param timeout: the timeout in seconds
        :return: a list of keys
        """

    @abc.abstractmethod
    def internal_kv_reset(self) -> None:
        """
        Resets the internal key-value store.
        This is useful for testing purposes and should not be used in production as it is a destructive operation.
        """

    @abc.abstractmethod
    def debug_info(self) -> str:
        """Returns debug information about this client."""
