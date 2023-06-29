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
    """Decorator to authenticate a request by authorization header"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            payload = await verify_user(self.request)
        except (web.HTTPUnauthorized, AssertionError):
            return web.json_response({"error": "Unauthorized"}, status=401)

        return await func(self, payload, *args, **kwargs)

    return wrapper

def authenticate_by_appkey(func):
    """Decorator to authenticate a request

    will check if the appkey is in the list of valid appkeys,
    and if it is, will return the user info as the first argument to the function
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        apikey: str = self.request.headers.get("Authorization", "")
        apikey = apikey.removeprefix("Bearer ")

        userinfo = prd.OPENAI_PRIVATE_EMBEDDINGS_API_KEYS.get(apikey)
        if not userinfo:
            raise web.HTTPUnauthorized()

        return await func(self, userinfo, *args, **kwargs)

    return wrapper


def recover(func):
    """Decorator to recover from exceptions"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            return web.HTTPBadRequest(text=str(e))

    return wrapper
