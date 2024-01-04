import functools
import hashlib
import time
from typing import Mapping

import aiohttp.web
import jwt

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
        raise aiohttp.web.HTTPUnauthorized()

    payload = jwt.decode(token, prd.SECRET_KEY, algorithms=["HS256"])
    assert payload["exp"] > time.time(), "jwt expired"

    return payload


def authenticate(func):
    """Decorator to authenticate a request by authorization header"""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        payload = await verify_user(self.request)
        return await func(self, payload, *args, **kwargs)

    return wrapper


def get_user_by_appkey(request: aiohttp.web.Request) -> prd.UserPermission:
    apikey: str = request.headers.get("Authorization", "")
    apikey = apikey.removeprefix("Bearer ")
    assert apikey, "apikey is required"

    uid: str = request.headers.get("X-Laisky-User-Id", "")
    assert isinstance(uid, str), "uid must be a string"

    api_base: str = (
        request.headers.get("X-Laisky-Openai-Api-Base", "") or "https://api.openai.com/"
    )
    assert isinstance(api_base, str), "api_base must be a string"

    model: str = request.query.get("model", "") or "gpt-3.5-turbo-1106"

    userinfo = prd.UserPermission(
        is_paid=False,
        uid=uid or hashlib.sha1(apikey.encode("utf-8")).hexdigest(),
        n_concurrent=100,
        chat_model=model,
        apikey=apikey,
        api_base=api_base.strip("/") + "/v1",
    )

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
        userinfo = get_user_by_appkey(self.request)
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
            return aiohttp.web.HTTPBadRequest(text=str(e))

    return wrapper
