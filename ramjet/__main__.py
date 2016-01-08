import logging
import traceback

import tornado.httpserver
import tornado.options as opt
from tornado.options import options

from ramjet.app import Application
from ramjet.settings import LOG_NAME
from ramjet.tasks import detect_tasks, _TASKS


logger = logging.getLogger(LOG_NAME)


def main():
    opt.parse_command_line()

    # 暂不需要 Web 应用
    # http_server = tornado.httpserver.HTTPServer(Application())
    # http_server.listen(options.port)

    if options.debug:
        logger.info('start application in debug mode')
        logger.setLevel(logging.DEBUG)
    else:
        logger.info('start application in normal mode')
        logger.setLevel(logging.INFO)
    ioloop = tornado.ioloop.IOLoop.instance()

    detect_tasks()
    for task in _TASKS:
        try:
            logger.info('start {}'.format(task.name))
            task.func()
        except Exception:
            logger.warn('run {} got error: '.format(task.name, traceback.format_exc()))

    ioloop.start()

if __name__ == '__main__':
    main()
