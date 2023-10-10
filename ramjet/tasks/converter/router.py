import aiohttp_jinja2
from aiohttp import web
from ramjet.utils import logger

logger = logger.getChild("converter")


def bind_handle(add_route):
    logger.info("bind_handle")
    add_route("/", Index)
    add_route("/mongoid/", MongoIDConverter)


class Index(web.View):
    @aiohttp_jinja2.template("converter/index.html")
    async def get(self):
        return


class MongoIDConverter(web.View):
    @aiohttp_jinja2.template("converter/mongo_id.html")
    async def get(self):
        return
