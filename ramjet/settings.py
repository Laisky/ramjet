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
    'heart',  # !!! 默认的心跳服务，不要注释
    # -----------------------------------
    'keyword',
]
