import logging
import pathlib
import tempfile
import multiprocessing

# server
HOST = "0.0.0.0"
PORT = 37851


MAIL_HOST = ""
MAIL_USERNAME = ""
MAIL_PASSWD = ""
MAIL_PORT = 0
MAIL_FROM_ADDR = ""
MAIL_TO_ADDRS = ""

# worker
N_PROCESS_WORKER = multiprocessing.cpu_count()
N_THREAD_WORKER = 20
print(f"Using {N_PROCESS_WORKER} process workers and {N_THREAD_WORKER} thread workers")

# common
CWD = pathlib.PurePath(__file__).parents[1]
LOG_NAME = "ramjet-driver"
LOG_DIR = tempfile.gettempdir()
LOG_PATH = "{}.log".format(pathlib.Path(LOG_DIR, LOG_NAME))
PID_FILE_PATH = "{}.lock".format(pathlib.Path(LOG_DIR, LOG_NAME))
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("ldap3").setLevel(logging.WARNING)
logging.getLogger("tweepy").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logger = logging.getLogger(LOG_NAME)

# web
OK = 0
ERROR = 1
URL_PREFIX = ""
SECRET_KEY = "ilori2y8KdbWVgsIf9eb39K29vy9zkHxelHihazxF2E="

# tasks
INSTALL_TASKS = [
    # 可以用简单的字符串定义任务，指向 tasks 里的文件夹名
    # tasks/*/__init__.py 里必须定义 bind_task
    # Warning: 默认的心跳服务，不要注释
    "heart",
    # 或者可以用 dict 详细定义
    # @parameter task: 任务名；
    # @parameter entry: 入口函数；
    # @parameter http_handle: HTTP 入口
    {"task": "webdemo", "entry": "bind_task", "http_handle": "bind_handle"},
    "webapps",
    "test",
    # -------------------------------------------------
    # 从下面开始是自定制的任务
    # -------------------------------------------------
    "keyword",
    "twitter",
    "aws",
    "email_proxy",
    "dlp",
    "upload",
    "monitors",
    "gptchat",
    "converter",
]


# db
MONGO_HOST = ""
MONGO_PORT = 0
MONGO_DB = ""
MONGO_USER = ""
MONGO_PASSWD = ""
