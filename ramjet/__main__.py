import logging

import click
from aiohttp import web

from ramjet.utils import Options, logger
from ramjet.app import setup_web_handlers
from ramjet import settings


@click.command()
@click.option('-t', '--tasks', default='', help='Tasks you want to run')
@click.option('-e', '--exclude-tasks', default='', help='Tasks you do not want to run')
@click.option('--debug', default=False, type=bool)
def main(**kw):
    options = Options()
    options.set_options(**kw)
    options.set_settings(**settings.__dict__)
    if options.get_option('debug'):
        logger.info('start application in debug mode')
        logger.setLevel(logging.DEBUG)
    else:
        logger.info('start application in normal mode')
        logger.setLevel(logging.INFO)

    from ramjet.tasks import setup_tasks

    app = web.Application()
    setup_tasks(app)
    setup_web_handlers(app)
    web.run_app(app, host='localhost', port=options.get_option('port'))

if __name__ == '__main__':
    main()
