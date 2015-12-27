import logging

import tornado.httpserver
import tornado.options as opt
from tornado.options import options

from ramjet.app import Application
from ramjet.settings import LOG_NAME


log = logging.getLogger(LOG_NAME)


def main():
    opt.parse_command_line()

    # 暂不需要 Web 应用
    # http_server = tornado.httpserver.HTTPServer(Application())
    # http_server.listen(options.port)

    if options.debug:
        log.info('start application in debug mode')
        log.setLevel(logging.DEBUG)
    else:
        log.info('start application in normal mode')
        log.setLevel(logging.INFO)
    ioloop = tornado.ioloop.IOLoop.instance()

    from ramjet.tasks import setup_tasks
    setup_tasks(ioloop)

    ioloop.start()

if __name__ == '__main__':
    main()
