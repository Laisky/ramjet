"""
Ramjet
"""

import hashlib
import logging

import aiohttp_jinja2
import jinja2
from aiohttp import web
from aiohttp_session import setup
from aiohttp_session.cookie_storage import EncryptedCookieStorage

from ramjet.settings import LOG_NAME, SECRET_KEY
from ramjet.utils import logger


class PageNotFound(web.View):
    async def get(self):
        return web.Response(status=404, text="404: not found!😢")


def setup_web_handlers(app):
    key = hashlib.md5(SECRET_KEY.encode("utf8")).hexdigest().encode("utf8")
    setup(app, EncryptedCookieStorage(key))
    web.view("/404.html", PageNotFound)


def setup_templates(app):
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader("./tasks"))
