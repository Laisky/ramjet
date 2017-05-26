"""
Web HTTP Hanle 的示例

访问：/apps/
"""
from aiohttp import web
from aiohttp_session import get_session
import aiohttp_jinja2

from ramjet.settings import logger


logger = logger.getChild('tasks.webapps')


def bind_task():
    logger.info("run webapps")


def bind_handle(add_route):
    logger.info('bind_handle')
    add_route('/', WebApps)


class WebApps(web.View):

    @aiohttp_jinja2.template('static/dist/index.html')
    async def get(self):
        logger.info('get WebApps')

        return None
