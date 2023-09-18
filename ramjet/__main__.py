import logging
import os
from pathlib import Path

import aiohttp_jinja2
import jinja2
from aiohttp import web
from kipp.options import opt
from kipp.utils import EmailSender

from ramjet import settings
from ramjet.app import setup_web_handlers
from ramjet.utils import logger


def setup_template(app):
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(str(Path(settings.CWD, "tasks", "templates"))),
    )


def setup_args():
    for k, v in settings.__dict__.items():
        opt.set_option(k, v)

    opt.add_argument("-t", "--tasks", default="", help="Tasks you want to run")
    opt.add_argument(
        "-e", "--exclude-tasks", default="", help="Tasks you do not want to run"
    )
    opt.add_argument("--debug", action="store_true", default=False)
    opt.add_argument("--smtp_host", type=str, default=None)
    opt.parse_args()


def setup_options():
    opt.set_option(
        "email_sender",
        EmailSender(
            host=settings.MAIL_HOST,
            port=settings.MAIL_PORT,
            username=settings.MAIL_USERNAME,
            passwd=settings.MAIL_PASSWD,
        ),
    )


async def health(req):
    return web.Response(text="hello, world")


def main():
    try:
        setup_args()
        setup_options()

        # is_allow_to_running = check_is_allow_to_running(settings.PID_FILE_PATH)
        # if not is_allow_to_running:
        #     print('another process is still running! exit now ...')
        #     return

        if opt.debug:
            logger.info("start application in debug mode")
            # logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            logger.info("start application in normal mode")
            logger.setLevel(logging.INFO)

        from ramjet.tasks import setup_tasks

        app = web.Application(
            client_max_size=100 * 1024**3,  # 100MB
        )
        app.router.add_get("/health", health)
        setup_tasks(app)
        setup_template(app)
        setup_web_handlers(app)
        logger.info(f"start web server {opt.HOST}:{opt.PORT}")
        web.run_app(app, host=opt.HOST, port=opt.PORT, keepalive_timeout=300)
    except Exception:
        logger.exception("ramjet got error:")
        # opt.email_sender.send_email(
        #     mail_to=settings.MAIL_TO_ADDRS,
        #     mail_from=settings.MAIL_FROM_ADDR,
        #     subject="ramjet error",
        #     content=traceback.format_exc(),
        # )
        raise
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
