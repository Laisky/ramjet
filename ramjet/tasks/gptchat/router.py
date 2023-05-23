import urllib.parse

import aiohttp.web

from .base import logger
from .embedding import setup, query

def bind_handle(add_route):
    logger.info("bind gpt web handlers")
    setup()

    add_route("", Index)
    add_route("query", Query)


class Index(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")

class Query(aiohttp.web.View):
    async def get(self):
        project = self.request.query.get("p")
        question = urllib.parse.unquote(self.request.query.get("q"))

        if not project:
            return aiohttp.web.Response(text="p is required", status=400)
        if not question:
            return aiohttp.web.Response(text="q is required", status=400)

        resp = await query(project, question)
        return aiohttp.web.json_response(resp._asdict())
