import pymongo

from ramjet.settings import MONGO_HOST, MONGO_PORT


def get_conn():
    return pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
