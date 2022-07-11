import aiohttp
import aiohttp_jinja2
from aiographql.client import GraphQLRequest
from ramjet.utils import get_conn, get_gq_cli


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
                    url
                    created_at
                    images
                    user {{
                        name
                    }}
                }}
            }}
        """
        )
        resp = await self.gq.query(query)
        data = resp.data
        assert data, f"load tweet `{tweet_id}` got: {resp.errors}"

        docu = data["TwitterStatues"][0]
        docu["images"] = docu.get("images", [])

        # load threads
        # query = GraphQLRequest(
        #     query=f"""
        #     query {{
        #         TwitterThreads(
        #             tweet_id: "{tweet_id}",
        #         ) {{
        #             id
        #             text
        #             url
        #             created_at
        #             user {{
        #                 name
        #             }}
        #         }}
        #     }}
        # """
        # )
        # resp = await self.gq.query(query)
        # assert resp.data, f"load tweet `{tweet_id}` threads: {resp.errors}"
        # threads = resp.data["TwitterThreads"]

        # import ipdb
        # ipdb.set_trace()

        return {
            "id": tweet_id,
            "text": docu["text"],
            "url": docu["url"],
            "image": docu["images"][0] if docu["images"] else "",
            "images": docu["images"],
            "created_at": docu["created_at"],
            "user": (docu.get("user", {}) or {}).get("name", "佚名"),
            # "threads": threads,
        }


class SearchStatus(BaseDisplay):
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
                    url
                    created_at
                    images
                    user {{
                        name
                    }}
                }}
            }}
        """
        )
        resp = await self.gq.query(query)
        data = resp.data
        assert data, f"search tweets `{search_text}` got: {resp.errors}"

        docus = list(filter(lambda d: d, data["TwitterStatues"]))
        return {
            "search_text": search_text,
            "tweets": docus,
        }
