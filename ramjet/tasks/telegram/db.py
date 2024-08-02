import pymongo

from ramjet.settings import prd


def get_db():
    mongo = pymongo.MongoClient(
        f"mongodb://{prd.MONGO_ADMIN_USER}:{prd.MONGO_ADMIN_PASSWD}@{prd.MONGO_HOST}:{prd.MONGO_PORT}",
    )
    return mongo["telegram"]
