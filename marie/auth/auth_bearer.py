from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED

from marie.auth.api_key_manager import APIKeyManager


class TokenBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(TokenBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        credentials = await HTTPBearer(auto_error=False)(request)
        if credentials is None or not APIKeyManager.is_valid(credentials.credentials):
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail='authentication_required',
                headers={'WWW-Authenticate': 'Bearer'},
            )
        return credentials.credentials
