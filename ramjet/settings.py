import os
import logging

# server
LISTEN_PORT = 27851

# worker
N_THREAD_WORKER = 8
N_PROCESS_WORKER = 8

# common
CWD = os.path.dirname(__file__)
LOG_NAME = 'ramjet-driver'
LOG_DIR = '/tmp'
LOG_PATH = '{}.log'.format(os.path.join(LOG_DIR, LOG_NAME))
logger = logging.getLogger(LOG_NAME)

# web
OK = 0
ERROR = 1

# tasks
INSTALL_TASKS = [
    # 可以用简单的字符串定义任务，指向 tasks 里的文件夹名
    # tasks/*/__init__.py 里必须定义 bind_task
    # Warning: 默认的心跳服务，不要注释
    'heart',
    # 或者可以用 dict 详细定义
    # @parameter task: 任务名；
    # @parameter entry: 入口函数；
    # @parameter http_handle: HTTP 入口
    {'task': 'web_demo', 'entry': 'bind_task', 'http_handle': 'setup_handle'},
    # -------------------------------------------------
    # 从下面开始是自定制的任务
    # -------------------------------------------------
    # 'keyword',
]
