import importlib

from ramjet import settings
from ramjet.utils import logger


def setup_tasks(app):

    def generate_add_route(task):
        def add_route(url, handle, method='*'):
            url = url.lstrip('/')
            app.router.add_route(method,
                                 '/{}/{}'.format(task, url),
                                 handle)

        return add_route

    for task in settings.INSTALL_TASKS:
        try:
            if isinstance(task, str):
                task = {'task': task}

            if isinstance(task, dict):
                m = importlib.import_module(
                    '.{}'.format(task['task']), 'ramjet.tasks')
                handle = getattr(m, task.get(
                    'http_handle', 'bind_handle'), None)
                handle and handle(generate_add_route(task['task']))
                entry = getattr(m, task.get('entry', 'bind_task'), None)
                entry and entry()
            else:
                raise "settings.INSTALL_TASKS syntax error"
        except Exception as err:
            logger.exception(err)
