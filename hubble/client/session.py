import requests

from ..utils import get_base_url
from .endpoints import EndpointsV2

__all__ = ['HubbleAPISession']


class HubbleAPISession(requests.Session):
    """The customized `requests.Session` object.

    ``HubbleAPISession`` helps the ``hubble.client.Client`` create a default
    ``header`` and validate the jwt token when calling ``init_jwt_auth``.

    The ``HubbleAPISession`` is initialized in the ``hubble.client.Client``
    constructor.
    """

    def __init__(self):
        super().__init__()

        self.headers.update(
            {
                'Accept-Charset': 'utf-8',
            }
        )

    def init_jwt_auth(self, token: str):
        """Initialize the jwt token.

        :param token: The api token user get from webpage.
        """
        self.headers.update({'Authorization': f'token {token}'})

    def validate_token(self) -> requests.Response:
        """Validate API token.

        This function will call the whoami endpoint from Hubble API
        to get user info.

        :return: a `requests.Response` object from Hubble API server.
        """
        url = get_base_url() + EndpointsV2.get_user_info
        resp = requests.post(url, headers=self.headers)
        resp.raise_for_status()

        return resp
