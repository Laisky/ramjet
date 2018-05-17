"""Task 示例
"""

import gc

from ramjet.settings import logger
from ramjet.engines import ioloop

logger = logger.getChild('tasks.heart')


def bind_task():
    def callback(*args, **kw):
        logger.info('tasks heart!')
        gc.collect()
        ioloop.call_later(60, callback, *args, **kw)

    ioloop.call_later(0, callback)
