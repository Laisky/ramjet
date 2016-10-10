"""Ramjet
"""

import logging

from tornado.options import define
from aiohttp import web

from ramjet.settings import LOG_NAME, LISTEN_PORT
from ramjet.utils import setup_log


log = logging.getLogger(LOG_NAME)
setup_log()
define('port', default=LISTEN_PORT, type=int)
define('debug', default=False, type=bool)


class PageNotFound(web.View):

    async def get(self):
        return web.Response(status=404, text="404: not found!")


def setup_web_handlers(app):
    app.router.add_route('*', '/404.html', PageNotFound)
