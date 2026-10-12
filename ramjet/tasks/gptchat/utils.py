import functools
import time
from typing import Mapping

import aiohttp.web
import jwt

from ramjet.settings import prd

from .base import logger
from .credentials import resolve_request_credentials
from .namespace import credential_scope


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
    """get_user_by_appkey binds new private scope to complete caller credentials.

    Wire UID/free assertions confer no ownership or paid capacity. All requests
    receive conservative admission until an authoritative grant exists.
    """
    try:
        credentials = resolve_request_credentials(
            request.headers.get("Authorization", ""),
            request.headers.get("X-Laisky-Openai-Api-Base", ""),
        )
    except ValueError as error:
        raise aiohttp.web.HTTPUnauthorized(reason=str(error)) from None
    apikey = credentials["api_key"]
    api_base = credentials["base_url"]
    model: str = request.query.get("model", "") or "gpt-4o-mini"

    userinfo = prd.UserPermission(
        is_paid=False,
        uid=credential_scope(apikey),
        n_concurrent=3,
        chat_model=model,
        apikey=apikey,
        api_base=api_base,
    )

    return userinfo


def authenticate_by_appkey(func):
    """Decorator to authenticate a request

    Derives a credential capability without asserting provider/account validity,
    and returns request-local user info as the first argument to the function
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        userinfo = get_user_by_appkey(self.request)
        return await func(self, userinfo, *args, **kwargs)

    return wrapper


def authenticate_by_appkey_sync(func):
    """Decorator to authenticate a request

    Derives a credential capability without asserting provider/account validity,
    and returns request-local user info as the first argument to the function
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
        except aiohttp.web.HTTPException:
            raise
        except Exception as e:
            logger.error("handler error (%s)", type(e).__name__)
            return aiohttp.web.HTTPBadRequest(text="Request could not be completed")

    return wrapper
