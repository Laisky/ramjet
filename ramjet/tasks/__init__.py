import importlib

from ramjet import settings
from ramjet.utils import logger, Options


options = Options()
_tasks = options.get_option('tasks')
tasks = _tasks and _tasks.split(',')
_exclude_tasks = options.get_option('exclude_tasks')
exclude_tasks = _exclude_tasks and _exclude_tasks.split(',')



def setup_webapp(app):
    app.router.add_static('/static', './ramjet/tasks/static/dist/', show_index=True)


def setup_tasks(app):
    setup_webapp(app)

    def generate_add_route(app, task):
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

            if isinstance(task, dict):
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

            else:
                raise "settings.INSTALL_TASKS syntax error"
        except Exception as err:
            logger.exception(err)
