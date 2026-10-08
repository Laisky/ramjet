"""
Ramjet
"""

import logging

import aiohttp_jinja2
import jinja2
from aiohttp import web
from aiohttp_session import setup
from aiohttp_session.cookie_storage import EncryptedCookieStorage

from ramjet import settings
from ramjet.settings import LOG_NAME, SECRET_KEY
from ramjet.settings.base import SECRET_KEY as DEFAULT_SECRET_KEY
from ramjet.oauth_state import create_cookie_key
from ramjet.utils import logger


class PageNotFound(web.View):
    async def get(self):
        return web.Response(status=404, text="404: not found!😢")


def setup_web_handlers(app):
    """setup_web_handlers validates the cookie secret and registers session storage."""
    key = create_cookie_key(
        getattr(settings, "SESSION_SECRET_KEY", SECRET_KEY), DEFAULT_SECRET_KEY
    )
    setup(app, EncryptedCookieStorage(key, httponly=True, samesite="Lax"))
    web.view("/404.html", PageNotFound)


def setup_templates(app):
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader("./tasks"))
