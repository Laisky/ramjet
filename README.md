Ramjet
===

[![versions](https://img.shields.io/badge/version-v1.0-blue.svg)]()
[![versions](https://img.shields.io/badge/license-MIT/Apache-blue.svg)]()

> わが征くは星の大海 —— Yang Wen-li

| ![](http://7xjvpy.dl1.z0.glb.clouddn.com/ramjet.jpg) |
|:--:|
| 后台脚本的引擎 |


## Install & Run

Need Python3.4.x.

```sh
$ python setup.py install
$ python -m ramjet
```


## Description

基于 Tornado 和 consurrent.futures 运行脚本（`tasks`）。

每一个 task 都需要实现接口 `bind_task(ioloop, thread_executor, process_executor)`，
利用 `ioloop`、`thread_executor`、`process_executor` 自行实现运行逻辑。


## Demo

### 异步

```py
def bind_task(ioloop, thread_executor, process_executor):
    # 将任务添加进事件循环中
    ioloop.add_future(async_task(), callback)


@tornado.gen.coroutine
def async_task():
    yield tornado.gen.sleep(3)
    yield async_child_task()


@tornado.gen.coroutine
def async_child_task():
    yield tornado.gen.sleep(1)
    print('child task ok!')


def callback(future):
    print('ok')

```

### 多线程 & 多进程

需要注意子进程没法回收，所以请确保 task 可以很好的结束。

```py
def bind_task(ioloop, thread_executor, process_executor):
    # 多线程
    thread_executor.submit(task, your_arguments)
    # 多进程
    process_executor.submit(task, your_arguments)


def task(*args, **kw):
    pass

```

### 定时任务

```
def bind_task(ioloop, thread_executor, process_executor):
    now = ioloop.time()
    run_at = now + 3600
    ioloop.call_at(run_at, task, your_auguments)


def task(*args, **kw):
    # 可以在 task 内设置下一次执行的时间
    # ioloop.run_at(run_at, task, *args, **kw)
```

## Versions

[更新日志](http://git01.dds.com/ops/ramjet/blob/master/docs/versions.md)
