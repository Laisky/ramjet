import pymongo
from aiographql.client import GraphQLClient

from ramjet.settings import MONGO_DB, MONGO_HOST, MONGO_PASSWD, MONGO_PORT, MONGO_USER


# only support twitter
def get_db():
    return pymongo.MongoClient(
        f"mongodb://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
    )


# deprecated: use get_db instead
get_conn = get_db


def get_gq_cli():
    return GraphQLClient(endpoint="https://gq.laisky.com/query/")
