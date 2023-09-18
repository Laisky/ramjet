import binascii
import datetime
import pickle
import random
import re
import string

import pytz

from .db import get_conn, get_gq_cli, get_db
from .encrypt import generate_passwd, generate_token, validate_passwd, validate_token
from .jinja import TemplateRendering, debug_wrapper
from .mail import send_alert
from .cache import Cache
from .log import logger

__all__ = [
    "get_conn",
    "get_gq_cli",
    "get_db",
    "generate_passwd",
    "generate_token",
    "validate_passwd",
    "validate_token",
    "TemplateRendering",
    "debug_wrapper",
    "send_alert",
    "Cache",
    "logger",
]

UTC = pytz.timezone("utc")
CST = pytz.timezone("Asia/Shanghai")


def obj2str(obj):
    return binascii.b2a_base64(pickle.dumps(obj)).decode("utf-8")


def str2obj(string):
    return pickle.loads(binascii.a2b_base64(string.encode("utf-8")))


def format_dt(dt):
    return datetime.datetime.strftime(dt, "%Y年%m月%d日 %H时%M分")


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


def validate_email(email):
    epat = re.compile(
        r"^[_a-z0-9-]+(\.[_a-z0-9-]+)*" r"@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$"
    )
    return epat.match(email)


def validate_mobile(mobile):
    ippat = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    return ippat.match(mobile)


def generate_random_string(length):
    alphbet = string.ascii_letters + "".join([str(i) for i in range(10)])
    return "".join([random.choice(alphbet) for i in range(length)])


def format_sec(s):
    hr = round(s // 3600, 2)
    minu = round(s % 3600 // 60, 2)
    sec = round(s % 60, 2)
    return "{}小时 {}分钟 {}秒".format(hr, minu, sec)
