"""
hrm_ocr.api.middleware.request_id
=================================
Middleware to inject a unique UUID4 request ID into the ASGI scope and response headers.
"""
from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        # Attach to request state for downstream access
        request.state.request_id = request_id
        
        response = await call_next(request)
        
        # Attach to response headers
        response.headers["X-Request-ID"] = request_id
        return response
