import aiohttp
import aiohttp_jinja2
import pymongo
from aiographql.client import GraphQLClient, GraphQLRequest
from ramjet.utils import get_conn, get_gq_cli, logger

from .base import replace_media_urls


class BaseDisplay(aiohttp.web.View):
    _gq = get_gq_cli()
    _mongo = get_conn()

    @property
    def col(self):
        return self._mongo["twitter"]["tweets"]

    @property
    def db(self):
        return self._mongo["twitter"]

    @property
    def gq(self):
        return self._gq


class Status(BaseDisplay):
    @aiohttp_jinja2.template("twitter/tweet.html")
    async def get(self):
        tweet_id = self.request.match_info["tweet_id"]
        query = GraphQLRequest(
            query=f"""
            query {{
                TwitterStatues(
                    tweet_id: "{tweet_id}",
                ) {{
                    text
                    created_at
                    images
                    user {{
                        name
                    }}
                }}
            }}
        """
        )
        docu = (await self.gq.query(query)).data["TwitterStatues"][0]
        docu["images"] = docu.get("images", [])

        # load threads
        query = GraphQLRequest(
            query=f"""
            query {{
                TwitterThreads(
                    tweet_id: "{tweet_id}",
                ) {{
                    id
                    text
                    created_at
                    user {{
                        name
                    }}
                }}
            }}
        """
        )
        threads = (await self.gq.query(query)).data["TwitterThreads"]

        return {
            "id": tweet_id,
            "text": docu["text"],
            "image": docu["images"][0] if docu["images"] else "",
            "images": docu["images"],
            "created_at": docu["created_at"],
            "user": (docu.get("user", {}) or {}).get("name", "佚名"),
            "threads": threads,
        }


class StatusSearch(BaseDisplay):
    @aiohttp_jinja2.template("twitter/search.html")
    async def get(self):
        return

    @aiohttp_jinja2.template("twitter/search.html")
    async def post(self):
        search_text = (await self.request.post())["text"]
        query = GraphQLRequest(
            query=f"""
            query {{
                TwitterStatues(
                    regexp: "{search_text}",
                    sort: {{sort_by: "created_at", order: DESC}},
                    page: {{size: 100, page: 0}},
                ) {{
                    id
                    text
                    images
                    user {{
                        name
                    }}
                }}
            }}
        """
        )
        docus = (await self.gq.query(query)).data["TwitterStatues"]
        docus = list(filter(lambda d: d, docus))
        return {
            "search_text": search_text,
            "tweets": docus,
        }
