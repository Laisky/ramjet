"""
Web HTTP Hanle 的示例

访问：/webdemo/
"""
from aiohttp import web

from ramjet.settings import logger


logger = logger.getChild('tasks.web_demo')


def bind_task():
    logger.info("run web_demo")


def bind_handle(app):
    logger.info('bind_handle')
    app.router.add_route('*', '/webdemo/', DemoHandle)


class DemoHandle(web.View):

    async def get(self):
        return web.Response(text="New hope")
