Ramjet
===

[![versions](https://img.shields.io/badge/version-v1.7-blue.svg)]()
[![PyPI version](https://badge.fury.io/py/ramjet.svg)](https://badge.fury.io/py/ramjet)
[![versions](https://img.shields.io/badge/license-MIT/Apache-blue.svg)]()
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> わが征くは星の大海 —— Yang Wen-li

| ![](http://7xjvpy.dl1.z0.glb.clouddn.com/ramjet.jpg) |
|:--:|
| 后台脚本的引擎 |


## Links

  - [Documents](http://laisky.github.io/ramjet/)
  - [Github](https://github.com/Laisky/ramjet)
  - [PyPI](https://pypi.python.org/pypi/ramjet)


## Install & Run

Need Python3.5.x.

```sh
# Install from pypi

$ pip install ramjet
```

```sh
# Install from source

$ python setup.py install
$ python -m ramjet [--debug=true]
```


## Description

基于 asyncio 和 consurrent.futures 运行脚本（`tasks`）。

每一个 task 都需要实现接口 `bind_task()`。

利用 `ioloop`、`thread_executor`、`process_executor` 自行实现运行逻辑。


## Demo

### 异步

```py
import random
import asyncio

from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    # 将任务添加进事件循环中
    asyncio.ensure_future(async_task())


async def async_task():
    await asyncio.sleep(3)
    for i in range(10):
        asyncio.ensure_future(async_child_task(i))


async def async_child_task(n):
    await asyncio.sleep(random.random())
    print('child task {} ok!'.format(n))

```

### 多线程 & 多进程

需要注意子进程没法回收，所以请确保 task 可以很好的结束。

```py
from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    # 多线程
    thread_executor.submit(task, your_arguments)
    # 多进程
    process_executor.submit(task, your_arguments)


def task(*args, **kw):
    pass

```

### 定时任务

```py
from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    delay = 3600
    ioloop.call_later(delay, task, your_auguments)


def task(*args, **kw):
    # 可以在 task 内设置下一次执行的时间
    # ioloop.call_later(delay, task, *args, **kw)
```

### HTTP

```py
from aiohttp import web

from ramjet.settings import logger


logger = logger.getChild('tasks.web_demo')


def bind_task():
    logger.info("run web_demo")


def bind_handle(add_route):
    add_route('/', DemoHandle)


class DemoHandle(web.View):

    async def get(self):
        return web.Response(text="New hope")
```

## Versions

[更新日志](https://github.com/Laisky/ramjet/blob/master/CHANGELOG.md)
