import importlib

from ramjet import settings
from ramjet.utils import logger


def setup_tasks(app):
    for task in settings.INSTALL_TASKS:
        try:
            if isinstance(task, str):
                importlib.import_module('.{}'.format(task), 'ramjet.tasks').bind_task()
            elif isinstance(task, dict):
                m = importlib.import_module('.{}'.format(task['task']), 'ramjet.tasks')
                handle = getattr(m, task.get('http_handle'))
                handle and handle(app)
                entry = getattr(m, task.get('entry', 'bind_task'))
                entry and entry()
            else:
                raise "settings.INSTALL_TASKS syntax error"
        except Exception as err:
            logger.exception(err)
