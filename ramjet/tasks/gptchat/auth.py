import functools
import time
from typing import Mapping

from aiohttp import web
import jwt
from ramjet.settings import prd


async def verify_user(request) -> Mapping:
    """Validate cookie and return payload

    Raises:
        web.HTTPUnauthorized: if cookie is not found
        AssertionError: if jwt is expired
    """
    token = request.headers.get("Authorization", "")
    token = token.removeprefix("Bearer ")

    if not token:
        raise web.HTTPUnauthorized()

    payload = jwt.decode(token, prd.SECRET_KEY, algorithms=["HS256"])
    assert payload["exp"] > time.time(), "jwt expired"

    return payload


def authenticate(func):
    """Decorator to authenticate a request"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            payload = await verify_user(self.request)
        except (web.HTTPUnauthorized, AssertionError):
            return web.json_response({"error": "Unauthorized"}, status=401)

        return await func(self.request, payload, *args, **kwargs)

    return wrapper
