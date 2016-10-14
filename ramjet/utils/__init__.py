import re
import random
import datetime
import sys
import string
import logging
import pickle
import binascii

import pytz

from ramjet.settings import LOG_NAME, LOG_PATH
from .jinja import debug_wrapper, TemplateRendering
from .mail import send_mail
from .db import get_conn


logger = logging.getLogger(LOG_NAME)
log = logger
__all__ = [
    'utcnow', 'setup_log', 'validate_email', 'validate_mobile', 'generate_random_string',
    'debug_wrapper', 'TemplateRendering', 'logger', 'log',
    'send_mail', 'format_dt', 'format_utcdt', 'cstnow', 'now',
    'get_conn',
    'obj2str', 'str2obj',
]
UTC = pytz.timezone('utc')
CST = pytz.timezone('Asia/Shanghai')


def obj2str(obj):
    return binascii.b2a_base64(pickle.dumps(obj)).decode('utf8')


def str2obj(string):
    return pickle.loads(binascii.a2b_base64(string.encode('utf8')))


def format_dt(dt):
    return datetime.datetime.strftime(dt, '%Y年%m月%d日 %H时%M分')


def format_utcdt(dt):
    dt = dt + datetime.timedelta(hours=8)
    return format_dt(dt)


def cstnow():
    return utcnow().astimezone(tz=CST)

now = cstnow


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
    # log.addHandler(fh)


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


def format_sec(s):
    hr = round(s // 3600, 2)
    minu = round(s % 3600 // 60, 2)
    sec = round(s % 60, 2)
    return '{}小时 {}分钟 {}秒'.format(hr, minu, sec)
