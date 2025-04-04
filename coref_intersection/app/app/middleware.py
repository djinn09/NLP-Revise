import time
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ResponseTimeMiddleWare(BaseHTTPMiddleware):
    """Middleware to add a X-Process-Time header to responses.

    This middleware calculates the time taken to process a request
    and adds the duration to the response headers.

    Methods:
        dispatch(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            Processes the request and adds the X-Process-Time header.

    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request, calculate the processing time, and add it to the response headers.

        Args:
            request (Request): The request object.
            call_next (Callable[[Request], Awaitable[Response]]): The next middleware in line.

        Returns:
            Response: The response object with the X-Process-Time header added.

        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
