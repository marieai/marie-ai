from typing import Optional

from fastapi import HTTPException
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED

from marie.logging.predefined import default_logger as logger
from marie_server.auth.api_key_manager import APIKeyManager


class TokenBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(TokenBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super(
            TokenBearer, self
        ).__call__(request)
        try:
            token = credentials.credentials
            logger.debug(f"Verifying token => {token}")

            if credentials:
                if not credentials.scheme == "Bearer":
                    raise HTTPException(
                        status_code=HTTP_403_FORBIDDEN,
                        detail="Invalid authentication scheme.",
                    )
                if not APIKeyManager.is_valid(token):
                    raise HTTPException(
                        status_code=HTTP_401_UNAUTHORIZED,
                        detail="Invalid token or expired token.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return credentials.credentials
            else:
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN, detail="Invalid authorization code."
                )
        except Exception as e:
            if isinstance(e, HTTPException):
                if e.status_code in [HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED]:
                    raise e
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail=str(e))
