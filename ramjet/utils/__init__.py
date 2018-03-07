import re
import os
import random
import datetime
import sys
import string
import logging
from logging.handlers import RotatingFileHandler as RFHandler
import pickle
import binascii
import multiprocessing

import pytz
from kipp.options import opt

from ramjet.settings import LOG_NAME, LOG_PATH, MAIL_FROM_ADDR, MAIL_TO_ADDRS
from ramjet.engines import thread_executor
from .jinja import debug_wrapper, TemplateRendering
from .mail import send_mail
from .encrypt import generate_token, validate_token, generate_passwd, validate_passwd
from .db import get_conn


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


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton


class MultiProcessLogHandler(logging.Handler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.setup_queue()
        thread_executor.submit(self.run_dispatcher)

    def get_email_sender():
        return opt.email_sender

    def emit(self, record):
        try:
            ei = record.exc_info
            if ei:
                _ = self.format(record)  # just to get traceback text into record.exc_text
                record.exc_info = None  # not needed any more
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)

    def run_dispatcher(self):
        while 1:
            record = self.queue.get()
            if not record:
                return

            # logging.getLogger().handle(record)
            if record.levelno > logging.WARNING:
                thread_executor.submit(self.get_email_sender().send_email,
                    mail_to=MAIL_TO_ADDRS,
                    mail_from=MAIL_FROM_ADDR,
                    subject='Ramjet error alert',
                    content='{}\n{}'.format(record.message, record.exc_text))

    def setup_queue(self):
        self.queue = multiprocessing.Queue(-1)

    def __exit__(self):
        super().__exit__()
        self.queue.put_nowait(None)


def setup_log():
    logger = logging.getLogger(LOG_NAME)
    _format = '[%(asctime)s - %(levelname)s - %(name)s] %(message)s'
    formatter = logging.Formatter(_format)
    # set stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # set roll file
    rf = RFHandler(LOG_PATH, maxBytes=1024*1024*100, backupCount=3, delay=0.05)
    rf.setLevel(logging.DEBUG)
    rf.setFormatter(formatter)
    # log
    logger.setLevel(logging.ERROR)
    logger.addHandler(MultiProcessLogHandler())
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(ch)
    # log.addHandler(rf)

    return logger


logger = setup_log()
# logger.propagate = False


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
