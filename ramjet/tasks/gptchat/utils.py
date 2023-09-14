import functools
import time
from typing import Mapping
import hashlib

import aiohttp.web
import jwt
from aiohttp import web

from ramjet.settings import prd

from .base import logger


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


def get_user_by_appkey(request: aiohttp.web.Request) -> prd.UserPermission:
    try:
        apikey: str = request.headers.get("Authorization", "")
        apikey = apikey.removeprefix("Bearer ")
        assert apikey, "apikey is required"

        model: str = request.query.get("model", "") or "gpt-3.5-turbo"

        userinfo = prd.UserPermission(
            is_free=False,
            uid=hashlib.sha1(apikey.encode()).hexdigest(),
            n_concurrent=100,
            chat_model=model,
            apikey=apikey,
        )
    except Exception:
        return web.HTTPUnauthorized()

    return userinfo


def authenticate_by_appkey(func):
    """Decorator to authenticate a request

    will check if the appkey is in the list of valid appkeys,
    and if it is, will return the user info as the first argument to the function
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        userinfo = get_user_by_appkey(self.request)
        return await func(self, userinfo, *args, **kwargs)

    return wrapper


def authenticate_by_appkey_sync(func):
    """Decorator to authenticate a request

    will check if the appkey is in the list of valid appkeys,
    and if it is, will return the user info as the first argument to the function
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        apikey: str = self.request.headers.get("Authorization", "")
        apikey = apikey.removeprefix("Bearer ")

        userinfo = get_user_by_appkey(apikey)
        return func(self, userinfo, *args, **kwargs)

    return wrapper


def recover(func):
    """Decorator to recover from exceptions"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            logger.exception("handler error")
            return web.HTTPBadRequest(text=str(e))

    return wrapper


# def get_user_by_uid(uid: str) -> prd.UserPermission:
#     """Get user by uid

#     Args:
#         uid (str): uid of user

#     Returns:
#         prd.UserPermission: user info
#     """
#     for user in prd.OPENAI_PRIVATE_EMBEDDINGS_API_KEYS.values():
#         if user.uid == uid:
#             return user

#     logger.debug(f"uid {uid=} not found, using default user")
#     return prd.UserPermission(
#         is_free=True,
#         uid=uid,
#         n_concurrent=0,
#         chat_model="",
#     )
