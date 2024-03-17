"""
Web HTTP Hanle 的示例

访问：/webdemo/
"""
from aiohttp import web
from aiohttp_session import get_session
from ramjet.utils.log import logger

logger = logger.getChild("tasks.web_demo")


def bind_task():
    logger.info("run web_demo")


def bind_handle(add_route):
    logger.info("bind_handle")
    add_route("/", DemoHandle)


class DemoHandle(web.View):
    async def get(self):
        logger.info("get DemoHandle")

        s = await get_session(self.request)
        if "skey" in s:
            logger.info("session work ok")
        else:
            s["skey"] = "123"

        return web.Response(text="New hope")
