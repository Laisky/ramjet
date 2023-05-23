#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
from functools import wraps

from asyncio_extras import async_contextmanager
import aiomysql
# import motor
# import motor.motor_asyncio
import pymongo
from kipp.options import opt

from ramjet.libs import classproperty


class MySQLdbExceptionHandler:
    """Interface of the exceptions belong to MySQLdb"""

    __mysqldb_module = None

    def get_mysqldb_exception(self, name):
        if not self.__mysqldb_module:
            self.import_mysqldb()

        return getattr(self.__mysqldb_module, name)

    def import_mysqldb(self):
        self.__mysqldb_module = aiomysql


class BaseMySQLModel(MySQLdbExceptionHandler):
    _db_pool = None

    def __init__(self):
        self._db_pool = None

    @classmethod
    async def make_connection(cls):
        if cls._db_pool:
            return cls

        cls._db_pool = await aiomysql.create_pool(
            host=opt.DATABASES[cls._db_name]['HOST'],
            port=opt.DATABASES[cls._db_name].get('PORT', None),
            user=opt.DATABASES[cls._db_name]['USER'],
            password=opt.DATABASES[cls._db_name]['PASSWORD'],
            db=opt.DATABASES[cls._db_name]['NAME'],
            cursorclass=aiomysql.cursors.SSCursor,
            charset='utf8mb4'
        )
        return cls

    @classmethod
    @async_contextmanager
    async def get_conn(cls):
        """Get connection without autocommit"""
        await cls.make_connection()
        async with cls._db_pool.acquire() as conn:
            yield conn

    @classmethod
    @async_contextmanager
    async def get_cursor(cls):
        """Simple way to get cursor with autocommit"""
        await cls.make_connection()
        async with cls.get_conn() as conn:
            async with conn.cursor() as cursor:
                yield cursor
            await conn.commit()

    @classmethod
    async def close(cls):
        return cls._db_pool.close()


class BaseMongoModel(object):
    _CONNECTION = None
    _MONGO_CONNECTION = None
    _db_name = None

    @classmethod
    def oid(cls, sid):
        return pymongo.ObjectId(sid)

    @classmethod
    async def make_connection(cls):
        if cls._CONNECTION:
            return cls

        if cls._db_alias == 'movoto':
            args = {
                'host': opt.MONGODB['HOST'],
                'port': opt.MONGODB['PORT'],
            }
            auth_args= {
                'name': opt.MONGODB['USER'],
                'password': opt.MONGODB['PASSWORD'],
            }
            cls._db_name = opt.MONGODB['NAME']
        else:
            args = {
                'host': opt.MONGODBS[cls._db_alias]['HOST'],
                'port': opt.MONGODBS[cls._db_alias]['PORT'],
            }
            auth_args = {
                'name': opt.MONGODBS[cls._db_alias]['USER'],
                'password': opt.MONGODBS[cls._db_alias]['PASSWORD'],
            }
            cls._db_name = opt.MONGODBS[cls._db_alias]['NAME']

        # cls._CONNECTION = motor.motor_asyncio.AsyncIOMotorClient(**args)
        cls._MONGO_CONNECTION = pymongo.MongoClient(**args)
        await cls.auth(auth_args)
        return cls

    @classmethod
    async def auth(cls, auth_args):
        await cls.db.authenticate(**auth_args)

    @classproperty
    def db(cls):
        return cls.get_db()

    @classproperty
    def conn(cls):
        return cls.get_conn()

    @classproperty
    def mongo_conn(cls):
        return cls.get_mongo_conn()

    @classproperty
    def collection(cls):
        return cls.get_collection()

    @classproperty
    def mongo_db(cls):
        return cls.get_mongo_db()

    @classproperty
    def mongo_collection(cls):
        return cls.get_mongo_collection()

    @classmethod
    def get_conn(cls):
        assert getattr(cls, '_CONNECTION'), '_CONNECTION not defined!'
        return cls._CONNECTION

    @classmethod
    def get_mongo_conn(cls):
        assert getattr(cls, '_MONGO_CONNECTION'), '_MONGO_CONNECTION not defined!'
        return cls._MONGO_CONNECTION

    @classmethod
    def get_db(cls):
        assert hasattr(cls, '_db_name'), '_db_alias not defined!'
        return cls._CONNECTION[cls._db_name]

    @classmethod
    def get_collection(cls):
        assert hasattr(cls, '_collection'), '_collection not defined!'
        return cls.get_db()[cls._collection]

    @classmethod
    def get_mongo_db(cls):
        assert hasattr(cls, '_db_name'), '_db_alias not defined!'
        return cls._MONGO_CONNECTION[cls._db_name]

    @classmethod
    def get_mongo_collection(cls):
        assert hasattr(cls, '_collection'), '_collection not defined!'
        return cls.get_db()[cls._collection]

    @classmethod
    def filter(cls, query):
        return cls.get_collection().find(query)
