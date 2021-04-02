"""Task 示例
"""

import gc

from ramjet.engines import ioloop
from ramjet.settings import logger

logger = logger.getChild("tasks.heart")


def bind_task():
    def callback(*args, **kw):
        logger.info("tasks heart!")
        gc.collect()
        ioloop.call_later(60, callback, *args, **kw)

    ioloop.call_later(0, callback)
