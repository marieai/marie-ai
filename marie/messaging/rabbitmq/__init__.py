from typing import Dict

from marie.logging.predefined import default_logger
import asyncio

from marie.messaging.rabbitmq.client import BlockingPikaClient
from marie.utils.json import to_json
