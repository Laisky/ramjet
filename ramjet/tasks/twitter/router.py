from .base import logger
from .crawler import CrawlerView
from .login import LoginHandle, OAuthHandle


def bind_handle(add_route):
    logger.info("bind_handle")
    add_route("login/", LoginHandle)
    add_route("oauth/", OAuthHandle)
    add_route("fetch/{tweet_id}/", CrawlerView)
