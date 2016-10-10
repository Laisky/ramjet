from aiohttp import web

from ramjet.settings import logger


logger = logger.getChild('tasks.web_demo')


def bind_task():
    logger.info("run web_demo")


def setup_handle(app):
    app.router.add_route('*', '/webdemo/', DemoHandle)


class DemoHandle(web.View):

    async def get(self):
        return web.Response(text="New hope")
