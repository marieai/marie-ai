import os
from json import JSONDecodeError

from requests import Response

DOMAIN = 'https://api.hubble.jina.ai'
PROTOCOL = 'rpc'
VERSION = 'v2'


def get_base_url():
    """Get the base url based on environment"""

    domain = DOMAIN
    if 'JINA_HUBBLE_REGISTRY' in os.environ:
        domain = os.environ['JINA_HUBBLE_REGISTRY']

    return f'{domain}/{VERSION}/{PROTOCOL}/'


def get_json_from_response(resp: Response) -> dict:
    """
    Get the JSON data from the response.
    If the response isn't JSON, the response information is lost.
    The error message must include the response body and status code.
    """
    try:
        return resp.json()
    except JSONDecodeError as err:
        raise JSONDecodeError(
            f'Response: {resp.text}, status code: {resp.status_code}; {err.msg}',
            err.doc,
            err.pos,
        ) from err
