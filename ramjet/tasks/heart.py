"""Task 示例
"""

from ramjet.settings import logger
from ramjet.engine import ioloop


logger = logger.getChild('tasks.heart')

TASK_NAME = 'heartbeats'


def bind_task():
    def callback(*args, **kw):
        logger.info('tasks heart!')
        ioloop.call_later(10, callback, *args, **kw)

    ioloop.call_later(0, callback)
