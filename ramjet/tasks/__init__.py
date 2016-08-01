import importlib

from ramjet import settings
from ramjet.utils import logger


def setup_tasks(ioloop=False):
    for app in settings.INSTALL_TASKS:
        try:
            importlib.import_module('.{}'.format(app), 'ramjet.tasks').bind_task()
        except Exception as err:
            logger.exception(err)
