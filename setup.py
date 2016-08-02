#!/usr/bin/env python

try:
    import setuptools
    from setuptools import setup
except ImportError:
    setuptools = None
    from distutils.core import setup
from pip.req import parse_requirements
from pip.download import PipSession

import re
import ramjet


requires = [str(i.req) for i in parse_requirements('requirements.txt',
                                                   session=PipSession())
            if i.req is not None]


def update_readme_version(version):
    ver_reg = re.compile(
        '(https://img\.shields\.io/badge/version-v'
        '[0-9]+\.[0-9]+(\.[0-9]+)?((dev|rc)[0-9]+)?'
        '-blue\.svg)'
    )
    _v = 'https://img.shields.io/badge/version-v{}-blue.svg'.format(version)
    with open('README.md', 'r') as f:
        src = f.read()

    with open('README.md', 'w') as f:
        dest = ver_reg.sub(_v, src)
        f.write(dest)


version = ramjet.__version__
update_readme_version(version)

kwargs = {}
kwargs['long_description'] = """
Ramjet
======

|versions| |PyPI version| |versions| |Commitizen friendly|

    わが征くは星の大海 —— Yang Wen-li

+------------------+
| |image4|         |
+==================+
| 后台脚本的引擎   |
+------------------+

Links
-----

-  `Documents <http://laisky.github.io/ramjet/>`__
-  `Github <https://github.com/Laisky/ramjet>`__
-  `PyPI <https://pypi.python.org/pypi/ramjet>`__

Install & Run
-------------

Need Python3.5.x.

.. code:: sh

    # Install from pypi

    $ pip install ramjet

.. code:: sh

    # Install from source

    $ python setup.py install
    $ python -m ramjet [--debug=true]

Description
-----------

基于 asyncio 和 consurrent.futures 运行脚本（\ ``tasks``\ ）。

每一个 task 都需要实现接口 ``bind_task()``\ 。

利用 ``ioloop``\ 、\ ``thread_executor``\ 、\ ``process_executor``
自行实现运行逻辑。

Demo
----

异步
~~~~

.. code:: py

    from ramjet.engines import process_executor, thread_executor, ioloop


    def bind_task():
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

多线程 & 多进程
~~~~~~~~~~~~~~~

需要注意子进程没法回收，所以请确保 task 可以很好的结束。

.. code:: py

    from ramjet.engines import process_executor, thread_executor, ioloop


    def bind_task():
        # 多线程
        thread_executor.submit(task, your_arguments)
        # 多进程
        process_executor.submit(task, your_arguments)


    def task(*args, **kw):
        pass

定时任务
~~~~~~~~

::

    from ramjet.engines import process_executor, thread_executor, ioloop


    def bind_task():
        now = ioloop.time()
        run_at = now + 3600
        ioloop.call_at(run_at, task, your_auguments)


    def task(*args, **kw):
        # 可以在 task 内设置下一次执行的时间
        # ioloop.run_at(run_at, task, *args, **kw)

Versions
--------

`更新日志 <https://github.com/Laisky/ramjet/blob/master/CHANGELOG.md>`__

.. |versions| image:: https://img.shields.io/badge/version-v1.2.1-blue.svg
   :target:
.. |PyPI version| image:: https://badge.fury.io/py/ramjet.svg
   :target: https://badge.fury.io/py/ramjet
.. |versions| image:: https://img.shields.io/badge/license-MIT/Apache-blue.svg
   :target:
.. |Commitizen friendly| image:: https://img.shields.io/badge/commitizen-friendly-brightgreen.svg
   :target: http://commitizen.github.io/cz-cli/
.. |image4| image:: http://7xjvpy.dl1.z0.glb.clouddn.com/ramjet.jpg
"""


setup(
    name='ramjet',
    version=version,
    packages=['ramjet'],
    include_package_data=True,
    install_requires=requires,
    author='Laisky',
    author_email='ppcelery@gmail.com',
    description='Scripts Manager',
    url='https://github.com/Laisky/ramjet',
    license='MIT License',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Libraries',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=[
        'tornado'
    ],
    **kwargs
)
