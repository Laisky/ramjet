import pymongo
from aiographql.client import GraphQLClient
from ramjet.settings import (MONGO_DB, MONGO_HOST, MONGO_PASSWD, MONGO_PORT,
                             MONGO_USER)


# only support twitter
def get_conn():
    return pymongo.MongoClient(
        f"mongodb://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"
    )


def get_gq_cli():
    return GraphQLClient(endpoint="https://graphql.laisky.com/query/")
