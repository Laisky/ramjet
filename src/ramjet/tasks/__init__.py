import os
import glob
import traceback
from collections import namedtuple
from importlib import import_module

from ramjet.settings import logger, CWD


TASK = namedtuple('task', ['name', 'func'])
_TASKS = []


def register_task(task_name, task_func):
    logger.info('register_task for task_name {}'.format(task_name))

    task = TASK(name=task_name, func=task_func)
    _TASKS.append(task)


def detect_tasks():
    logger.info('detect_tasks')

    for taskfpath in glob.glob(os.path.join(CWD, 'tasks', '*')):
        taskfname = os.path.split(taskfpath)[1]
        if taskfname.startswith('__'):
            continue

        module_name = None
        if os.path.isfile(taskfpath):
            if not taskfpath.endswith('.py'):
                continue

            module_name = os.path.splitext(taskfname)[0]
            if module_name == '__init__':
                continue
        else:
            module_name = taskfname

        if not module_name:
            continue

        try:
            logger.debug('try to import ramjet.tasks.{}'.format(module_name))
            module = import_module('ramjet.tasks.{}'.format(module_name))
            task = module.bind_task
        except ImportError:
            logger.error('import module {} error: {}'
                         .format(module_name, traceback.format_exc()))
        except AttributeError:
            logger.error('bind_task not found in {}'.format(module))
        except Exception:
            pass
        else:
            task_name = getattr(task, 'TASK_NAME', module_name)
            register_task(task_name, task)

    return _TASKS
