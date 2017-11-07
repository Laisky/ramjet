
# simple test
if __name__ == '__main__':
    from ramjet.models import get_mysql_model, MovotoMongoModel
    from ramjet.engines import ioloop

    async def demo():
        MovotoDB = get_mysql_model('movoto')
        await MovotoDB.make_connection()
        async with MovotoDB.get_conn() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute('show databases;')
                print(await cursor.fetchall())
                conn.commit()

    ioloop.run_until_complete(demo())


    async def demo():
        await MovotoMongoModel.make_connection()
        db = MovotoMongoModel.db
        r = await db['run_stats_monitor'].find_one()
        print(r)

    ioloop.run_until_complete(demo())
