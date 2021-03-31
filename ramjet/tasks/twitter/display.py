import aiohttp
import aiohttp_jinja2
import pymongo
from ramjet.utils import get_conn, logger

from .base import replace_media_urls


class BaseDisplay(aiohttp.web.View):
    @property
    def col(self):
        return get_conn()["twitter"]["tweets"]

    @property
    def db(self):
        return get_conn()["twitter"]


class Status(BaseDisplay):
    @aiohttp_jinja2.template("twitter/tweet.html")
    async def get(self):
        tweet_id = self.request.match_info["tweet_id"]
        docu = self.col.find_one({"id_str": f"{tweet_id}"})
        if not docu:
            return aiohttp.web.HTTPNotFound()

        replace_media_urls(docu)
        images = [
            media["media_url_https"]
            for media in docu.get("entities", {}).get("media", [])
        ]
        return {
            "id": tweet_id,
            "text": docu["text"],
            "image": "" if len(images) == 0 else images[0],
            "images": images,
            "user": docu.get("user", {}).get("name", "佚名"),
        }


class StatusSearch(BaseDisplay):
    @aiohttp_jinja2.template("twitter/search.html")
    async def get(self):
        return

    @aiohttp_jinja2.template("twitter/search.html")
    async def post(self):
        text = (await self.request.post())["text"]
        docus = (
            self.col.find({"text": {"$regex": text}})
            .sort("created_at", pymongo.DESCENDING)
            .limit(50)
        )
        return {
            "tweets": docus,
        }
