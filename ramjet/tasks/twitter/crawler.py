import json
from pathlib import Path
from threading import RLock
from typing import Dict

import pymongo
import requests
import tweepy
from aiohttp import web
from ramjet.engines import ioloop, thread_executor
from ramjet.settings import (ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY,
                             CONSUMER_SECRET, TWITTER_IMAGE_DIR)
from ramjet.utils import get_conn
from tweepy import API, OAuthHandler

from .base import gen_related_tweets, logger, twitter_api_parser

lock = RLock()


def bind_task():
    def run():
        later = 60
        ioloop.call_later(later, run)

        logger.info("run")
        twitter = TwitterAPI()
        thread_executor.submit(twitter.run)

    run()


class FetchView(web.View):
    async def get(self):
        tweet_id = self.request.match_info["tweet_id"]
        return web.Response(
            text=f"fetch specific tweet id {tweet_id}",
        )

    async def post(self):
        tweet_id = (await self.request.post())["tweet_id"]
        logger.info(f"fetch tweet {tweet_id}")
        thread_executor.submit(TwitterAPI().run_for_tweet_id, tweet_id)
        return web.Response(text=f"starting to fetch {tweet_id}")


class TwitterAPI:

    __api = None
    __auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    _current_user_id: int

    @property
    def api(self):
        logger.debug("get api")
        if self.__api:
            return self.__api

        return self.set_api()

    def set_api(
        self, access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET
    ):
        self.__auth.set_access_token(access_token, access_token_secret)
        self.__api = API(
            self.__auth, wait_on_rate_limit=True, parser=tweepy.parsers.JSONParser()
        )
        return self.__api

    def _save_replies(self, tweet: Dict[str, any]):
        try:
            for new_tweet in self.__api.search(
                q=f"to:{tweet['user']['screen_name']}",
                since_id=tweet["id"],
                tweet_mode="extended",
            )["statuses"]:
                try:
                    self.save_tweet(new_tweet)
                    self._save_replies(new_tweet)
                    # self._save_relate_tweets(new_tweet)
                except Exception:
                    logger.exception("save tweet")
        except tweepy.error.RateLimitError:
            time.sleep(10)
        except Exception:
            logger.exception("_save_replies")

    def g_load_tweets(self, last_id: int):
        """
        Twitter 只能反向回溯，从最近的推文开始向前查找
        """
        last_id = max(int(last_id) - 100, 0)
        # last_id = 0

        logger.info(f"g_load_tweets for {last_id=}")
        last_tweets = self.api.user_timeline(count=1, tweet_mode="extended")
        if not last_tweets:
            return

        yield last_tweets[0]
        current_id = last_tweets[0]["id"]
        while True:
            tweets = self.api.user_timeline(
                max_id=current_id, count=100, tweet_mode="extended"
            )
            if len(tweets) == 1:  # 到头了
                logger.info("loaded all tweets")
                return

            for s in tweets:
                if s["id"] <= last_id:  # 已存储
                    return

                yield s
                if s["id"] < current_id:
                    current_id = s["id"]

                for sid in gen_related_tweets(self.db["tweets"], s):
                    try:
                        s = self.api.get_status(sid, tweet_mode="extended")
                    except tweepy.error.TweepError as err:
                        logger.warn(f"get status got error: {err}")
                        continue
                    except Exception as err:
                        logger.exception(f"get status {sid}")
                        continue

                    yield s

    @property
    def col(self):
        return get_conn()["twitter"]["tweets"]

    @property
    def db(self):
        return get_conn()["twitter"]

    def parse_tweet(self, tweet):
        logger.debug("parse_tweet")
        return twitter_api_parser(tweet)

    def get_last_tweet_id(self):
        """
        获取数据库里存储的最后一条推文
        """
        logger.debug("get_last_tweet_id")
        docu = self.db["tweets"].find_one(
            {"user.id": self._current_user_id}, sort=[("id", pymongo.DESCENDING)]
        )
        return docu and docu["id"]

    def save_tweet(self, tweet: Dict[str, any]):

        # parse tweet
        docu = self.parse_tweet(tweet)
        self.download_images_for_tweet(tweet)

        # save tweet
        logger.info(f"save_tweet {docu['id']}")
        self.db["tweets"].update_one(
            {"id_str": str(docu["id"])},
            {"$set": docu, "$addToSet": {"viewer": self._current_user_id}},
            upsert=True,
        )
        user = docu.get("user")
        if user:
            user["id_str"] = str(user["id"])
            self.db["users"].update_one(
                {"id_str": str(user["id"])}, {"$set": user}, upsert=True
            )

    def _convert_media_url(self, src: str) -> str:
        src = src.replace(
            "http://pbs.twimg.com/media/", "https://s3.laisky.com/uploads/twitter/"
        )
        src = src.replace(
            "https://pbs.twimg.com/media/", "https://s3.laisky.com/uploads/twitter/"
        )
        return src

    def _download_image(self, tweet: Dict[str, any], media_entity: Dict[str, any]):
        tweet["text"] = tweet["text"].replace(
            media_entity["url"],
            self._convert_media_url(media_entity["media_url_https"]),
        )

        fpath = Path(TWITTER_IMAGE_DIR, media_entity["media_url_https"].split("/")[-1])
        if fpath.is_file():
            return

        with requests.get(media_entity["media_url_https"] + ":orig") as r:
            if r.status_code != 200:
                logger.error(
                    f"download {media_entity['media_url_https']}: [{r.status_code}]{r.content}"
                )
                return

            with open(fpath, "wb") as f:
                f.write(r.content)

            logger.info(f"succeed download image {tweet['id']} -> {fpath}")

    def _download_video(self, tweet: Dict[str, any], media_entity: Dict[str, any]):
        max_bitrate = 0
        max_url = ""
        for v in media_entity["video_info"]["variants"]:
            if v.get("bitrate", 0) > max_bitrate:
                max_bitrate = v.get("bitrate", 0)
                max_url = v.get("url", "")

        if not max_url:
            return

        max_url = max_url[: max_url.rfind("?")]
        tweet["text"] = tweet["text"].replace(
            media_entity["url"], self._convert_media_url(max_url)
        )

        fpath = Path(TWITTER_IMAGE_DIR, max_url.split("/")[-1])
        if fpath.is_file():
            return

        with requests.get(max_url) as r:
            if r.status_code != 200:
                logger.error(f"download {max_url}: [{r.status_code}]{r.content}")
                return

            with open(fpath, "wb") as f:
                f.write(r.content)

            logger.info(f"succeed download image {tweet['id']} -> {fpath}")

    def download_images_for_tweet(self, tweet: Dict[str, any]):
        media = tweet.get("extended_entities", {}).get("media", []) or tweet.get(
            "entities", {}
        ).get("media", [])

        for img in media:
            if img["type"] == "photo":
                self._download_image(tweet, img)
            elif img["type"] == "video":
                self._download_video(tweet, img)

    def g_load_user(self):
        logger.debug("g_load_user_auth")

        for u in self.db["account"].find():
            yield u

    def _save_relate_tweets(self, status: Dict[str, any]):
        try:
            related_ids = []
            status.get("in_reply_to_status_id") and related_ids.append(
                status["in_reply_to_status_id"]
            )
            status.get("retweeted_status", {}).get("id") and related_ids.append(
                status["retweeted_status"]["id"]
            )
            status.get("quoted_status", {}).get("id") and related_ids.append(
                status["quoted_status"]["id"]
            )
            related_ids = filter(
                lambda id_: not self.db["tweets"].find_one({"id": id_}), related_ids
            )

            for id_ in related_ids:
                try:
                    docu = self.api.get_status(id_, tweet_mode="extended")
                except tweepy.error.RateLimitError:
                    time.sleep(10)
                except Exception:
                    logger.exception(f"load tweet {id_} got error")
                    raise
                else:
                    logger.info(
                        f"save tweet [{docu['user']['screen_name']}]{docu['id']}"
                    )
                    self.save_tweet(docu)
                    self._save_relate_tweets(docu)
        except Exception:
            logger.exception(f"_save_relate_tweets")

    def run_for_tweet_id(self, tweet_id: str):
        for u in self.g_load_user():
            self._current_user_id = u["id"]
            self.set_api(u["access_token"], u["access_token_secret"])

            try:
                tweet = self.api.get_status(tweet_id, tweet_mode="extended")
                assert tweet
            except Exception:
                logger.exception(f"load tweet {tweet_id} got error")
                continue
            else:
                self.save_tweet(tweet)
                self._save_relate_tweets(tweet)
                self._save_replies(tweet)
                return

    def run(self):
        if not lock.acquire(blocking=False):
            return

        logger.debug("run TwitterAPI")
        count = 0
        try:
            for u in self.g_load_user():
                try:
                    logger.info(f"fetch tweets for user {u['username']}")
                    self._current_user_id = u["id"]
                    self.set_api(u["access_token"], u["access_token_secret"])

                    last_id = self.get_last_tweet_id() or 1
                    for count, status in enumerate(self.g_load_tweets(last_id)):
                        self._save_relate_tweets(status)
                        self._save_replies(status)
                        self.save_tweet(status)
                except Exception:
                    logger.exception(f"try load user {u['username']} tweets")
        except Exception as err:
            logger.exception(err)
        else:
            logger.info("save {} tweets".format(count))
        finally:
            lock.release()

    def run_for_archive_data(self, user_id: int, tweet_fpath: str):
        self._current_user_id = user_id
        user = self.db["users"].find_one({"id_str": str(user_id)})
        del user["_id"]

        with open(tweet_fpath) as fp:
            for tweet in json.loads(fp.read()):
                tweet = tweet["tweet"]
                tweet["user"] = user
                self._save_relate_tweets(tweet)
                self._save_replies(tweet)
                self.save_tweet(tweet)
