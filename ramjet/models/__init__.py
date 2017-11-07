"""
=============
Simple Models
=============

Examples:

MySQL Movotodb
::
    from ramjet.engines import ioloop
    from ramjet.models import MovotoDB

    # without autocommit
    async def demo():
        async with MovotoDB.get_conn() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('show databases;')
                print(await cursor.fetchall())
                await conn.commit()

    # with autocommit
    async def demo():
        async with MovotoDB.cursor() as cursor:
            await cursor.execute('show databases;')
            print(await cursor.fetchall())
            await conn.commit()

    ioloop.run_until_complete(demo())


Mongo Movoto
::
    from ramjet.engines import ioloop
    from ramjet.models import MovotoMongoModel

    async def demo():
        await MovotoMongoModel.make_connection()
        db = MovotoMongoModel.db

        # get single document
        r = await db['run_stats_monitor'].find_one()
        print(r)

        # get more than one documents
        cursor = db['run_stats_monitor'].find()
        cursor.sort('_id', -1).limit(50)
        async for document in curosr:
            # do something with document

    ioloop.run_until_complete(demo())

"""

from .mongo_dbs import MovotoMongoModel
from .mysql_dbs import get_mysql_model, MovotoDB
