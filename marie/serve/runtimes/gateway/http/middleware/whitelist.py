from typing import Optional
from fastapi import Request, HTTPException, status
from marie.logging.logger import MarieLogger


class WhitelistMiddleware:
    def __init__(self, logger: "MarieLogger", ip_whitelist: Optional[list] = None):
        self.logger = logger
        self.ip_whitelist = ip_whitelist

    async def __call__(self, request: Request, call_next):
        if request.client.host not in self.ip_whitelist:
            self.logger.warning(
                f"Non-whitelist request from host {request.client.host}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )

        response = await call_next(request)
        return response
