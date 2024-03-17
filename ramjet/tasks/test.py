import aiohttp
import aiohttp_jinja2
from ramjet.utils.log import logger

logger = logger.getChild("tasks.test")


def bind_handle(add_route):
    logger.info("bind_handle")
    add_route("/", TestView)


class TestView(aiohttp.web.View):
    @aiohttp_jinja2.template("test.html")
    async def get(self):
        return
