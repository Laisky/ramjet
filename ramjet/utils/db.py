import pymongo


def get_conn():
    return pymongo.MongoClient('localhost', 27016)
