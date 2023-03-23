import aiohttp.web

from .base import logger

def bind_handle(add_route):
    logger.info("bind gpt web handlers")
    add_route("", Index)


class Index(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")
