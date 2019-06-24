import pathlib
import importlib

from aiohttp import web

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
        def add_route(url, handler, method='*'):
            path = f"{settings.URL_PREFIX}/{task}/{url.lstrip('/')}"
            if issubclass(handler, web.View):
                logger.info(f'bind http class based view for path: {path}')
                # app.router.add_route("*", path, handler)  # aiohttp < 3
                app.router.add_view(path, handler)  # aiohttp >= 3
            else:
                assert method in ['get', 'post', 'delete', 'head', 'option', 'put']
                logger.info(f"bind http handler view for method {method}, path: {path}")
                getattr(app.router, f"add_{method}")(path, handler)
        return add_route

    for taskcfg in settings.INSTALL_TASKS:
        try:
            if isinstance(taskcfg, str):
                taskcfg = {'task': taskcfg}

            # check -t
            if tasks:
                if taskcfg['task'] not in tasks:
                    continue

            # check -e
            if exclude_tasks:
                if taskcfg['task'] in exclude_tasks:
                    continue

            if not isinstance(taskcfg, dict):
                raise "settings.INSTALL_TASKS syntax error"

            m = importlib.import_module(f".{taskcfg['task']}", 'ramjet.tasks')
            handler = getattr(m, taskcfg.get('http_handle', 'bind_handle'), None)
            if handler:
                logger.info('bind http handler: %s', taskcfg['task'])
                handler(generate_add_route(app, taskcfg['task']))

            entry = getattr(m, taskcfg.get('entry', 'bind_task'), None)
            if entry:
                logger.info('bind handler: %s', taskcfg['task'])
                entry()

        except Exception as err:
            logger.exception(err)
