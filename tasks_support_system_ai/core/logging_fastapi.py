"""Logging for fastapi middleware.

Overpowered version is here (I don't like logging body requests and responses)
https://medium.com/@dhavalsavalia/fastapi-logging-middleware-logging-requests-and-responses-with-ease-and-style-201b9aa4001a
"""

import logging
import traceback

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, logger: logging.Logger) -> None:
        self._logger = logger
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.query_params:
            path += f"?{request.query_params}"

        try:
            response = await call_next(request)
            status_code = response.status_code

            self._logger.info(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "path": path,
                    "ip": request.client.host,
                    "res_status_code": status_code,
                }
            )
            return response

        except Exception as e:
            for handler in self._logger.handlers:
                # use standard uvicorn logger in console
                if not isinstance(handler, logging.StreamHandler) or isinstance(
                    handler, logging.FileHandler
                ):
                    self._logger.error(
                        {
                            "method": request.method,
                            "url": str(request.url),
                            "path": path,
                            "ip": request.client.host,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                            "res_status_code": 500,
                        },
                        exc_info=True,
                    )
            raise
