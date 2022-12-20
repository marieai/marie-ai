import io
import json
from typing import Dict, List, Optional, Union

import requests

from ..utils.api_utils import get_json_from_response
from .base import BaseClient
from .endpoints import EndpointsV2


class Client(BaseClient):
    def create_personal_access_token(
        self, name: str, expiration_days: int = 30
    ) -> Union[requests.Response, dict]:
        """Create a personal access token.

        Personal Access Token (refer as PAT) is same as `token`
        where you get from the UI.
        The main difference is that you can set a ``expiration_days``
        for PAT while ``token`` becomes invalid as soon as user logout.

        :param name: The name of the personal access token.
        :param expiration_days: Number of days to be valid, by default 30 days.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.create_pat,
            data={'name': name, 'expirationDays': expiration_days},
        )

    def list_personal_access_tokens(self) -> Union[requests.Response, dict]:
        """List all created personal access tokens.

        All expired PATs will be automatically deleted.
        The list function only shows valid PATs.

        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(url=self._base_url + EndpointsV2.list_pats)

    def delete_personal_access_token(self, name: str) -> Union[requests.Response, dict]:
        """Delete personal access token by name.

        :param name: Name of the personal access token
          to be deleted.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.delete_pat,
            data={'name': name},
        )

    def get_user_info(self, log_error: bool = True) -> Union[requests.Response, dict]:
        """Get current logged in user information.

        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.get_user_info, log_error=log_error
        )

    @property
    def token(self):
        try:
            self.get_user_info(log_error=False)
            return self._token
        except Exception:
            return None

    @property
    def username(self) -> str:
        user = self.get_user_info(log_error=False).get('data', {})
        return user.get('nickname') or user.get('name')

    def upload_artifact(
        self,
        f: Union[str, io.BytesIO],
        id: Optional[str] = None,
        metadata: Optional[dict] = None,
        is_public: bool = False,
        show_progress: bool = False,
    ) -> Union[requests.Response, dict]:
        """Upload artifact to Hubble Artifact Storage.

        :param f: The full path or the `io.BytesIO` of the file to be uploaded.
        :param id: Optional value, the id of the artifact.
        :param metadata: Optional value, the metadata of the artifact.
        :param is_public: Optional value, if this artifact is public or not,
          default not public.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        from rich import filesize

        from ..utils.pbar import get_progressbar

        self.get_user_info()  # to make sure the user is logged in.

        pbar = get_progressbar(disable=not show_progress)

        class BufferReader(io.BytesIO):
            def __init__(self, buf=b''):
                super().__init__(buf)
                self._len = len(buf)

                self.task = pbar.add_task(
                    'Uploading',
                    total=self._len,
                    start=True,
                    total_size=str(filesize.decimal(self._len)),
                )

            def __len__(self):
                return self._len

            def read(self, n=-1):
                chunk = io.BytesIO.read(self, n)
                pbar.update(self.task, advance=len(chunk))
                return chunk

        if isinstance(f, str):
            f = open(f, 'rb')
        elif not isinstance(f, io.BytesIO):
            raise TypeError(
                f'Unexpected type {type(f)}, expect either `str` or `io.BytesIO`.'
            )

        dict_data = {
            'public': is_public,
            'file': ('file', f.read()),
        }

        if id:
            dict_data['id'] = id
        if metadata:
            dict_data['metaData'] = json.dumps(metadata)

        (data, ctype) = requests.packages.urllib3.filepost.encode_multipart_formdata(
            dict_data
        )

        headers = {'Content-Type': ctype}

        with pbar:
            return self.handle_request(
                url=self._base_url + EndpointsV2.upload_artifact,
                data=BufferReader(data),
                headers=headers,
            )

    def download_artifact(
        self, id: str, f: Union[str, io.BytesIO], show_progress: bool = False
    ) -> str:
        """Download artifact from Hubble Artifact Storage to localhost.

        :param id: The id of the artifact to be downloaded.
        :param f: The full path or the `io.BytesIO` of the file to be downloaded.
        :returns: A str object indicates the download path on localhost or bytes.
        """
        from rich import filesize

        from ..utils.pbar import get_progressbar

        # first get download uri.
        resp = self.handle_request(
            url=self._base_url + EndpointsV2.download_artifact,
            data={'id': id},
        )

        # Second download artifact.
        if isinstance(resp, requests.Response):
            resp = get_json_from_response(resp)
        download_url = resp['data']['download']

        pbar = get_progressbar(disable=not show_progress)

        with pbar:
            with requests.get(download_url, stream=True) as response:
                total = int(response.headers.get('content-length'))
                task = pbar.add_task(
                    'Downloading',
                    total=total,
                    start=True,
                    total_size=str(filesize.decimal(total)),
                )

                if isinstance(f, str):
                    with open(f, 'wb') as writer:
                        for data in response.iter_content(chunk_size=1024 * 1024):
                            writer.write(data)
                            pbar.update(task, advance=len(data))
                elif isinstance(f, io.BytesIO):
                    for data in response.iter_content(chunk_size=1024 * 1024):
                        f.write(data)
                        pbar.update(task, advance=len(data))
                else:
                    raise TypeError(
                        f'Unexpected type {type(f)}, expect either'
                        '`str` or `io.BytesIO`.'
                    )
        return f

    def delete_artifact(
        self, id: Optional[str] = None, name: Optional[str] = None
    ) -> Union[requests.Response, dict]:
        """Delete the artifact from Hubble Artifact Storage.

        :param id: The id of the artifact to be deleted.
        :param name: The name of the artifact to be deleted.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.delete_artifact,
            data={'id': id} if id else {'name': name},
        )

    def delete_multiple_artifacts(
        self, *, ids: Optional[List[str]] = None, names: Optional[List[str]] = None
    ) -> Union[requests.Response, dict]:
        """Delete multiple artifacts from Hubble Artifact Storage.

        :param ids: A list of the IDs of the artifacts to be deleted.
        :param names: A list of the names of the artifacts to be deleted.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        data = {}
        if ids:
            data['ids'] = ids
        if names:
            data['names'] = names

        return self.handle_request(
            url=self._base_url + EndpointsV2.delete_multiple_artifacts,
            data=data,
        )

    def get_artifact_info(
        self, id: Optional[str] = None, name: Optional[str] = None
    ) -> Union[requests.Response, dict]:
        """Get the metadata of the artifact.

        :param id: The id of the artifact.
        :param name: The name of the artifact.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.get_artifact_info,
            data={'id': id} if id else {'name': name},
        )

    def update_artifact(
        self,
        id: str,
        *,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
        is_public: Optional[bool] = None,
    ) -> Union[requests.Response, dict]:
        """Update artifact.

        :param id: The id of the artifact to be updated.
        :param name: Optional, a new name.
        :param metadata: Optional, a new metadata.
        :param public: Optional, change visibility to public or private.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """

        data = {
            'id': id,
            'name': name,
            'metaData': json.dumps(metadata) if metadata else None,
            'public': is_public,
        }

        return self.handle_request(
            url=self._base_url + EndpointsV2.update_artifact,
            json={key: value for (key, value) in data.items() if value is not None},
        )

    def list_artifacts(
        self,
        *,
        filter: Optional[dict] = None,
        sort: Optional[Dict[str, int]] = None,
        pageIndex: Optional[int] = None,
        pageSize: Optional[int] = None,
    ) -> Union[requests.Response, List[dict]]:
        """Get list of artifacts.

        :param filter: optional, to filter by fields.
        :param sort: optional, to sort by fields.
        :param pageIndex: optional, to specify which page to load.
        :param pageSize: optional, number of items per page.
        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """

        data = {
            'filter': filter,
            'sort': sort,
            'pageIndex': pageIndex,
            'pageSize': pageSize,
        }

        return self.handle_request(
            url=self._base_url + EndpointsV2.list_artifacts,
            json={key: value for (key, value) in data.items() if value is not None},
        )

    def list_internal_docker_registries(
        self,
    ) -> Union[requests.Response, dict]:
        """List internal docker registries.

        :returns: `requests.Response` object as returned value
            or indented json if jsonify.
        """
        return self.handle_request(
            url=self._base_url + EndpointsV2.list_internal_docker_registries,
        )
