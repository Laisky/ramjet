import aiohttp
import aiohttp_jinja2

from .base import logger
from .crawler import FetchView
from .display import Status, StatusSearch
from .login import LoginHandle, OAuthHandle


def bind_handle(add_route):
    logger.info("bind_handle")
    add_route("login/", LoginHandle)
    add_route("oauth/", OAuthHandle)
    add_route("fetch/", FetchView)
    add_route("status/search/", StatusSearch)
    add_route("status/{tweet_id}/", Status)
    add_route("", Index)


class Index(aiohttp.web.View):
    @aiohttp_jinja2.template("twitter/index.html")
    async def get(self):
        return
