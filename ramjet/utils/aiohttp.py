import functools

from aiohttp import web

from .common import logger


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
