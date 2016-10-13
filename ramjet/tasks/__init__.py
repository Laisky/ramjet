import importlib

from ramjet import settings
from ramjet.utils import logger


def setup_tasks(app):
    for task in settings.INSTALL_TASKS:
        try:
            if isinstance(task, str):
                task = {'task': task}

            if isinstance(task, dict):
                m = importlib.import_module('.{}'.format(task['task']), 'ramjet.tasks')
                handle = getattr(m, task.get('http_handle', 'bind_handle'), None)
                handle and handle(app)
                entry = getattr(m, task.get('entry', 'bind_task'), None)
                entry and entry()
            else:
                raise "settings.INSTALL_TASKS syntax error"
        except Exception as err:
            logger.exception(err)
