import re
import random
import datetime
import sys
import string
import logging

import pytz

from ramjet.settings import LOG_NAME, LOG_PATH
from .jinja import debug_wrapper, TemplateRendering


log = logging.getLogger(LOG_NAME)
__all__ = [
    'utcnow', 'setup_log', 'validate_email', 'validate_mobile', 'generate_random_string',
    'debug_wrapper', 'TemplateRendering',
]
UTC = pytz.timezone('utc')


def utcnow():
    return datetime.datetime.utcnow().replace(tzinfo=UTC)


def setup_log():
    _format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(_format)
    # set stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # set log file
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    # log
    logging.getLogger(LOG_NAME).setLevel(logging.DEBUG)
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(ch)
    # log.addHandler(sh)
    log.addHandler(fh)


def validate_email(email):
    epat = re.compile(r'^[_a-z0-9-]+(\.[_a-z0-9-]+)*'
                      r'@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$')
    return epat.match(email)


def validate_mobile(mobile):
    ippat = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    return ippat.match(mobile)


def generate_random_string(length):
    alphbet = string.ascii_letters + ''.join([str(i) for i in range(10)])
    return ''.join([random.choice(alphbet) for i in range(length)])
