"""
Ramjet
"""

import logging
import base64

from tornado.options import define
from aiohttp import web
from aiohttp_session.cookie_storage import EncryptedCookieStorage
from aiohttp_session import setup

from ramjet.settings import LOG_NAME, LISTEN_PORT, SECRET_KEY
from ramjet.utils import setup_log


log = logging.getLogger(LOG_NAME)
setup_log()
define('port', default=LISTEN_PORT, type=int)
define('debug', default=False, type=bool)


class PageNotFound(web.View):

    async def get(self):
        return web.Response(status=404, text="404: not found!")


def setup_web_handlers(app):
    secret_key = base64.urlsafe_b64decode(SECRET_KEY)
    setup(app, EncryptedCookieStorage(secret_key))
    app.router.add_route('*', '/404.html', PageNotFound)
