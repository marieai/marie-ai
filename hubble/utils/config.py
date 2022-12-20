import json
import os
from pathlib import Path

CONFIG_FILE_NAME = 'config.json'
ROOT_ENV_NAME = 'JINA_HUB_ROOT'


class Config:
    """
    This class is used to store the configuration of the application.
    """

    def __init__(
        self,
        config_file_name: str = CONFIG_FILE_NAME,
        root_env_name: str = ROOT_ENV_NAME,
    ):
        self.root = Path(os.environ.get(root_env_name, os.path.expanduser('~/.jina')))
        self.config_file = self.root.joinpath(config_file_name)

        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str = None):
        """
        Get the value of the key from the config.

        :param key: The key of the config. If it's None, then return the whole config.
        """
        if not self.config_file.exists():
            return None

        with open(self.config_file) as f:
            config = json.load(f)
            if key is None:
                return config
            else:
                return config.get(key)

    def set(self, key: str, value: str):
        """
        Set the value of the key to the config.

        :param key: The key of the config.
        :param value: The value of the config.

        :return: Whole config.
        """
        config = self.get()
        if config is None:
            config = {}

        config[key] = value
        with open(self.config_file, 'w') as f:
            json.dump(config, f)

        return config

    def delete(self, key: str):
        """
        Delete the key from the config.

        :param key: The key of the config.

        :return: Whole config.
        """
        config = self.get()
        if config is None:
            return None

        if key in config:
            del config[key]
            with open(self.config_file, 'w') as f:
                json.dump(config, f)

        return config

    def purge(self):
        """
        Purge the config.
        """
        if self.config_file.exists():
            self.config_file.unlink()


config = Config()
