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
with open('README.md', 'r') as f:
    kwargs['long_description'] = f.read()


setup(
    name='ramjet',
    version=version,
    packages=['ramjet'],
    include_package_data=True,
    install_requires=requires,
    author='Laisky',
    author_email='ppcelery@gmail.com',
    description='Scripts Manager',
    url='http://darth.chexiang.com',
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
    ],
    keywords=[
        'tornado'
    ],
    **kwargs
)
