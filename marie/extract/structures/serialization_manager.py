import pickle
from typing import Optional, Type, TypeVar

T = TypeVar("T")

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerializationManager:
    @staticmethod
    def serialize(data: T, file_path: str) -> None:
        """
        Serialize (pickle) the provided data and save it to the given file path.

        :param file_path: The file path where data will be stored.
        :param data: The Python object to serialize.
        """
        try:
            logger.debug("Starting serialization process.")
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
            logger.info("Serialization completed successfully.")
        except Exception as e:
            logger.error("Serialization failed with error: %s", e)
            raise e

    @staticmethod
    def deserialize(file_path: str, expected_type: Type[T]) -> Optional[T]:
        """
        Deserialize (unpickle) the data from the given file path and return it as the expected type.

        :param file_path: The file path from which data will be loaded.
        :param expected_type: The expected type of the data when deserialized.
        :return: The deserialized Python object of type T, or None if an error occurs.
        """

        logger.debug("Starting deserialization process.")
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            if not isinstance(data, expected_type):
                logger.warning(
                    f"Warning: Expected type {expected_type}, but got {type(data)}."
                )
            logger.info("Deserialization completed successfully.")
            return data
        except Exception as e:
            logger.error("Deserialization failed with error: %s", e)
            raise
