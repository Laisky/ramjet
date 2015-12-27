"""Task 示例
"""

from ramjet.settings import logger

logger = logger.getChild('tasks.heart')


def bind_task(ioloop, te, pe):
    def callback(*args, **kw):
        logger.info('tasks heart!')
        ioloop.call_later(10, callback, *args, **kw)

    ioloop.call_later(0, callback)
