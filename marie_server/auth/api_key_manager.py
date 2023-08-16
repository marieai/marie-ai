from typing import List, Dict, Any, Optional
from marie.logging.predefined import default_logger as logger

import secrets


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "y", "true", "t", "1")


class KeyGenerator:
    """
    Generates keys
    """

    @classmethod
    def generate_key(cls, prefix: str, length: Optional[int] = 54) -> str:
        """
        Generate a key
        :param prefix: the prefix to use for the key. This is useful for generating keys for different purposes.
        Two keys with the same prefix will not be the same.

        ref : https://stackoverflow.com/questions/13378815/base64-length-calculation
        :param length: the length of the key to generate
        :return: the generated key
        """

        nbytes = int((length * 3) / 4)
        generated_key = secrets.token_urlsafe(nbytes)
        return prefix + generated_key

    @classmethod
    def validate_key(cls, key: str) -> bool:
        """
        Check if a key is valid (i.e. has the correct length and prefix)
        :param key: the key to check
        :return: True if the key is valid, False otherwise
        """

        if len(key) != 58:
            return False

        if key[:4] not in ["mas_", "mau_"]:
            return False

        return True


class APIKeyManager:
    """
    Manages API keys for the server.
    """

    _keys = dict()

    @classmethod
    def add_key(cls, key_conf: dict[str, Any]) -> None:
        """
        Add a key to the manager
        :param key_conf: the key configuration to add
        """
        if "name" not in key_conf:
            raise ValueError("Key must have a 'name' property")

        if "api_key" not in key_conf:
            raise ValueError("Key must have a 'api_key' property")

        name = key_conf["name"]
        logger.info(f"Adding API key : {name}")

        enabled = True
        if "enabled" in key_conf:
            enabled = str2bool(key_conf["enabled"])

        if name in cls._keys:
            raise ValueError(f"Key with name '{name}' already exists")

        # validate that there are no duplicated keys
        for key in cls._keys:
            if cls._keys[key]["api_key"] == key_conf["api_key"]:
                raise ValueError(
                    f"Key with name '{name}' has the same api_key as key '{key}'"
                )

        # validate that the key is valid
        if not KeyGenerator.validate_key(key_conf["api_key"]):
            raise ValueError(
                f"Key with name '{name}' is not valid"
                f" (must be 58 characters long and start with 'mas_' or 'mau_')"
            )

        cls._keys[key_conf["api_key"]] = {
            "name": key_conf["name"],
            "api_key": key_conf["api_key"],
            "enabled": enabled,
        }

    @classmethod
    def get_keys(cls) -> dict[str, Any]:
        """
        Get the keys
        """
        return cls._keys

    @classmethod
    def is_valid(cls, key: str) -> bool:
        """
        Check if a key is valid
        :param key: the key to check
        :return: True if the key is valid and enabled, False otherwise
        """

        if key in cls._keys:
            if cls._keys[key]["enabled"]:
                return True
        return False

    @classmethod
    def from_config(cls, auth_config: Dict[str, Any]) -> None:
        """
        Create an APIKeyManager from a config
        Keys are in the format

        mau_ for Marie-AI App user-to-server tokens
        mas_ for Marie-AI App server-to-server tokens
        keys are of length 54 + 4(prefix) = 58, with a prefix of either mau_ or mas_
        ^mau_[a-zA-Z0-9]{54}$  or ^mas_[a-zA-Z0-9]{54}$

        Sample keys :
          mas_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ
          mas_xw_rZMMvT3snDw7IUQAe6FB3iChESG8Nn8Ek6riarHYRPl85BlW4vA
          mau_0aPJ9Q9nUO1Ac1vJTfffXEXs9FyGLf9BzfYgZ_RaHm707wmbfHJNPQ
          mau_Gcp_GvCMrVVgp-BwGKLyELE3BaKtpmCrlwdIB-VWWWXwpm3k1CwVIg
          mas_XeuXeznfHd_n0qRqavWSu9EVD0OrcwnJwvl_NOz0ucBG5R3creEWmw

        :param auth_config:
        :return:
        """
        logger.info(f"Setting up API keys from config")
        if "keys" in auth_config:
            for key in auth_config["keys"]:
                cls.add_key(key)
        else:
            logger.warning(f"No API keys found in config")

        logger.info(f"API keys setup : {len(cls.get_keys())}")
