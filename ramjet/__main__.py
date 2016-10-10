import logging

import tornado.options as opt
from tornado.options import options
from aiohttp import web


from ramjet.app import setup_web_handlers
from ramjet.settings import LOG_NAME


log = logging.getLogger(LOG_NAME)


def main():
    opt.parse_command_line()

    if options.debug:
        log.info('start application in debug mode')
        log.setLevel(logging.DEBUG)
    else:
        log.info('start application in normal mode')
        log.setLevel(logging.INFO)

    from ramjet.tasks import setup_tasks

    app = web.Application()
    setup_tasks(app)
    setup_web_handlers(app)
    web.run_app(app, host='localhost', port=options.port)

if __name__ == '__main__':
    main()
