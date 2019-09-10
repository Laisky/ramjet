import pymongo

from ramjet.settings import (
    MONGO_HOST, MONGO_PORT, MONGO_DB,
    MONGO_USER, MONGO_PASSWD
)


# only support twitter
def get_conn():
    return pymongo.MongoClient(f'mongodb://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}')
