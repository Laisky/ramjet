"""Ramjet
"""

import logging
from pathlib import Path

import tornado
from tornado.web import url
from tornado.options import define, options

from ramjet.settings import CWD, LOG_NAME, LISTEN_PORT
from ramjet.utils import setup_log, generate_random_string
from ramjet.views import BaseHandler


log = logging.getLogger(LOG_NAME)
setup_log()
define('port', default=LISTEN_PORT, type=int)
define('debug', default=False, type=bool)


class PageNotFound(BaseHandler):

    @tornado.gen.coroutine
    def get(self, url=None):
        log.debug('GET PageNotFound for url {}'.format(url))

        if url is None:
            self.render2('404.html', url=url)
            self.finish()
            return

        self.redirect_404()


class Application(tornado.web.Application):

    def __init__(self):
        settings = {
            'static_path': str(Path(CWD, 'static')),
            'static_url_prefix': '/static/',
            'template_path': str(Path(CWD, 'templates')),
            'cookie_secret': generate_random_string(50),
            'login_url': '/login/',
            'xsrf_cookies': True,
            'autoescape': None,
            'debug': options.debug
        }
        handlers = [
            # -------------- handler --------------
            url(r'^/404.html$', PageNotFound, name='404'),
        ]
        handlers.append(('/(.*)', PageNotFound))
        self.setup_db()
        self.setup_sentry()
        super(Application, self).__init__(handlers, **settings)
