Ramjet
===
<!-- markdownlint-disable MD003 MD042 MD045 MD007 -->

[![versions](https://img.shields.io/badge/version-v1.8.5-blue.svg)]()
[![PyPI version](https://badge.fury.io/py/ramjet.svg)](https://badge.fury.io/py/ramjet)
[![versions](https://img.shields.io/badge/license-MIT/Apache-blue.svg)]()
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> わが征くは星の大海

| ![](https://s3.laisky.com/uploads/2025/11/bussard_ramjet.jpeg) |
|:--:|
| The engine for cronjob scripts |


## Links

  - [Documents](http://laisky.github.io/ramjet/)
  - [Github](https://github.com/Laisky/ramjet)
  - [PyPI](https://pypi.python.org/pypi/ramjet)


## Install & Run

Need Python 3.10+.

```sh
# Install from PyPI

$ pip install ramjet
```

```sh
# Install from source for development

$ pip install --user pdm
$ pdm install
$ pdm run python -m ramjet [--debug=true]

# Optional: export pinned requirements for deployment-only environments
$ pdm export --without-hashes --format requirements -o requirements.txt
```


> Maintainers: run `pdm run update-readme-version` before tagging a release to refresh the README badge.


## Description

Run scripts (`tasks`) based on asyncio and concurrent.futures.

Each task needs to implement the `bind_task()` interface.

Use `ioloop`, `thread_executor`, and `process_executor` to implement your own execution logic.


## Demo

### Asynchronous

```py
import random
import asyncio

from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    # Add the task to the event loop
    asyncio.ensure_future(async_task())


async def async_task():
    await asyncio.sleep(3)
    for i in range(10):
        asyncio.ensure_future(async_child_task(i))


async def async_child_task(n):
    await asyncio.sleep(random.random())
    print('child task {} ok!'.format(n))

```

### Multithreading & Multiprocessing

Note: Subprocesses cannot be recycled, so please ensure that the task can terminate properly.

```py
from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    # Multithreading
    thread_executor.submit(task, your_arguments)
    # Multiprocessing
    process_executor.submit(task, your_arguments)


def task(*args, **kw):
    pass

```

### Scheduled Tasks

```py
from ramjet.engines import process_executor, thread_executor, ioloop


def bind_task():
    delay = 3600
    ioloop.call_later(delay, task, your_auguments)


def task(*args, **kw):
    # You can set the next execution time inside the task
    # ioloop.call_later(delay, task, *args, **kw)
```

### HTTP

```py
from aiohttp import web

from ramjet.utils.log import logger


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

[Changelog](https://github.com/Laisky/ramjet/blob/master/CHANGELOG.md)
