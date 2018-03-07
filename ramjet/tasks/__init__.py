import os
import pathlib
import importlib

from kipp.options import opt

from ramjet import settings
from ramjet.utils import logger


_tasks = getattr(opt, 'tasks', None)
tasks = _tasks and _tasks.split(',')
_exclude_tasks = getattr(opt, 'exclude_tasks', None)
exclude_tasks = _exclude_tasks and _exclude_tasks.split(',')



def setup_webapp(app):
    app.router.add_static('/static', pathlib.Path(settings.CWD, 'tasks', 'static').absolute(), show_index=True)


def setup_tasks(app):
    setup_webapp(app)

    def generate_add_route(app, task):
        task = task.replace('_', '-')
        def add_route(url, handle, method='*'):
            url = url.lstrip('/')
            app.router.add_route(method,
                                 '{}/{}/{}'.format(settings.URL_PREFIX, task, url),
                                 handle)
        return add_route

    for task in settings.INSTALL_TASKS:
        try:
            if isinstance(task, str):
                task = {'task': task}

            # check -t
            if tasks:
                if task['task'] not in tasks:
                    continue

            # check -e
            if exclude_tasks:
                if task['task'] in exclude_tasks:
                    continue

            if not isinstance(task, dict):
                raise "settings.INSTALL_TASKS syntax error"

            m = importlib.import_module(
                '.{}'.format(task['task']), 'ramjet.tasks')
            handle = getattr(m, task.get(
                'http_handle', 'bind_handle'), None)
            if handle:
                logger.info('bind http handle: %s', task['task'])
                handle(generate_add_route(app, task['task']))

            entry = getattr(m, task.get('entry', 'bind_task'), None)
            if entry:
                logger.info('bind handle: %s', task['task'])
                entry()

        except Exception as err:
            logger.exception(err)
