from functools import partial

from aiohttp import web
from kipp.utils import EmailSender
from ramjet.engines import ioloop, thread_executor
from ramjet.utils.log import logger as root_logger

logger = root_logger.getChild("tasks.email_proxy")


def bind_task():
    logger.info("run email proxy")


def bind_handle(add_route):
    logger.info("bind handle")
    add_route("/", EmailProxyHandle)


class EmailProxyHandle(web.View):
    async def get(self):
        return web.Response(text="email proxy")

    async def post(self):
        data = await self.request.json()
        sender = EmailSender(
            host=data.pop("host"),
            username=data.pop("username"),
            passwd=data.pop("passwd"),
            use_tls=data.pop("use_tls", None),
        )
        runner = partial(sender.send_email, **data)
        r = await ioloop.run_in_executor(thread_executor, runner)

        return web.Response(text=str(r))
