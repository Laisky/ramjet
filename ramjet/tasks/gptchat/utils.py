import functools

import aiohttp.web

from ramjet.settings import prd

from .base import logger
from .credentials import resolve_request_credentials


def get_user_by_appkey(request: aiohttp.web.Request) -> prd.UserPermission:
    """get_user_by_appkey returns model options without granting private ownership.

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
        uid="public",
        n_concurrent=3,
        chat_model=model,
        apikey=apikey,
        api_base=api_base,
    )

    return userinfo


def authenticate_by_appkey(func):
    """Decorator to authenticate a request

    Resolves model options without asserting provider/account validity,
    and returns request-local user info as the first argument to the function
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        userinfo = get_user_by_appkey(self.request)
        return await func(self, userinfo, *args, **kwargs)

    return wrapper


def authenticate_by_appkey_sync(func):
    """Decorator to authenticate a request

    Resolves model options without asserting provider/account validity,
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
