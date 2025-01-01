"""Logging for fastapi middleware.

Overpowered version is here (I don't like logging body requests and responses)
https://medium.com/@dhavalsavalia/fastapi-logging-middleware-logging-requests-and-responses-with-ease-and-style-201b9aa4001a
"""

import logging

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, logger: logging.Logger) -> None:
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        try:
            status_code = response.status_code
        except AttributeError:
            status_code = 500

        path = request.url.path
        if request.query_params:
            path += f"?{request.query_params}"

        self._logger.info(
            {
                "method": request.method,
                "url": str(request.url),
                "path": path,
                "ip": request.client.host,
                "res_status_code": status_code,
            },
        )
        return response
