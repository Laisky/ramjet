import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Union

import pymongo
import requests
import tweepy
from aiohttp import web
from ramjet.engines import ioloop, thread_executor
from ramjet.settings import (ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY,
                             CONSUMER_SECRET, S3_BUCKET, S3_KEY, S3_REGION,
                             S3_SECRET, S3_SERVER, TWITTER_IMAGE_DIR)
from ramjet.utils import get_conn
from tweepy import API, OAuthHandler

from .base import (gen_related_tweets, get_image_filepath, get_s3_key, logger,
                   replace_media_urls, parse_tweet_text)
from .s3 import connect_s3, is_file_exists, upload_file_in_mem

lock = RLock()


def bind_task():
    def run():
        later = 60
        ioloop.call_later(later, run)
        thread_executor.submit(_run)

        def _run():
            if not lock.acquire(blocking=False):
                return

            try:
                logger.info("run")
                twitter = TwitterAPI()
                twitter.crawl_timeline()
            except Exception as err:
                logger.exception("run twitter task")
            finally:
                lock.release()

    run()


class FetchView(web.View):
    async def get(self):
        tweet_id = self.request.match_info["tweet_id"]
        return web.Response(
            text=f"fetch specific tweet id {tweet_id}",
        )

    async def post(self):
        tweet_id = str((await self.request.post())["tweet_id"])
        tweet_id = tweet_id.strip("/")
        if "/" in tweet_id:
            tweet_id = tweet_id.split("/")[-1]

        if "?" in tweet_id:
            tweet_id = tweet_id.split("?")[0]

        logger.info(f"fetch tweet {tweet_id}")
        thread_executor.submit(TwitterAPI().run_for_tweet_id, tweet_id)
        return web.Response(text=f"succeed submit {tweet_id}")


class TwitterAPI:

    __api: API = None
    __auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    _current_user_id: int
    __s3cli = None

    def __init__(self):
        self.__s3cli = connect_s3(
            S3_SERVER,
            S3_REGION,
            S3_KEY,
            S3_SECRET,
        )

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

    def _save_replies(self, tweet: Dict[str, Any]):
        logger.debug(f"_save_replies for {tweet['id']}")
        try:
            for new_tweet in self.__api.search_tweets(
                q=f"to:{tweet['user']['screen_name']}",
                since_id=tweet["id"],
                tweet_mode="extended",
            )["statuses"]:
                if new_tweet["id"] == tweet["id"]:
                    continue

                try:
                    self.save_tweet(new_tweet)
                    # self._save_replies(new_tweet)
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
        last_tweets = self.api.user_timeline(
            count=1,
            tweet_mode="extended",
            exclude_replies=False,
            include_rts=True,
        )
        if not last_tweets:
            return

        yield last_tweets[0]
        current_id: int = last_tweets[0]["id"]
        is_last_batch: bool = False
        while True:
            tweets = self.api.user_timeline(
                max_id=current_id,
                count=100,
                exclude_replies=False,
                include_rts=True,
                tweet_mode="extended",
            )
            if len(tweets) == 1:  # 到头了
                logger.info("loaded all tweets")
                return

            for s in tweets:
                if s["id"] <= last_id:
                    # arrive at the newest tweet in db
                    is_last_batch = True

                yield s
                if s["id"] < current_id:
                    current_id = s["id"]

                # for sid in gen_related_tweets(self.db["tweets"], s):
                #     try:
                #         s = self.api.get_status(sid, tweet_mode="extended")
                #     except tweepy.error.TweepError as err:
                #         logger.warn(f"get status got error: {err}")
                #         continue
                #     except Exception as err:
                #         logger.exception(f"get status {sid}")
                #         continue

                #     yield s

            if is_last_batch:
                return

    @property
    def col(self):
        return get_conn()["twitter"]["tweets"]

    @property
    def db(self):
        return get_conn()["twitter"]

    def parse_tweet(self, tweet:Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("parse_tweet")
        return parse_tweet_text(tweet)

    def get_last_tweet_id(self):
        """
        获取数据库里存储的最后一条推文
        """
        logger.debug("get_last_tweet_id")
        docu = self.db["tweets"].find_one(
            {"user.id": self._current_user_id}, sort=[("id", pymongo.DESCENDING)]
        )
        return docu and docu["id"]

    def save_tweet(self, tweet: Dict[str, Any]):

        if self.col.find_one({"id_str": tweet["id_str"]}, projection=["_id"]):
            return

        # parse tweet
        tweet = self.parse_tweet(tweet)
        self.download_medias_for_tweet(tweet)

        # save tweet
        logger.info(f"save_tweet {tweet.get('id')} {tweet.get('created_at')}")
        self.db["tweets"].update_one(
            {"id_str": str(tweet["id"])},
            {"$set": tweet, "$addToSet": {"viewer": self._current_user_id}},
            upsert=True,
        )
        user = tweet.get("user")
        if user:
            user["id_str"] = str(user["id"])
            self.db["users"].update_one(
                {"id_str": str(user["id"])}, {"$set": user}, upsert=True
            )

    # def _convert_media_url(self, src: str) -> str:
    #     src = src.replace(
    #         "http://pbs.twimg.com/media/", "https://s3.laisky.com/uploads/twitter/"
    #     )
    #     src = src.replace(
    #         "https://pbs.twimg.com/media/", "https://s3.laisky.com/uploads/twitter/"
    #     )
    #     return src

    def _download_image(
        self, tweet: Dict[str, Any], media_entity: Dict[str, Any]
    ) -> List[str]:
        fkey = get_s3_key(media_entity)
        media_url = f"https://s3.laisky.com/{S3_BUCKET}/{fkey}"
        with requests.get(media_entity["media_url_https"] + ":orig") as r:
            assert (
                r.status_code == 200
            ), f"download {media_entity['media_url_https']}: [{r.status_code}]{r.content}"

            if is_file_exists(self.__s3cli, len(r.content), S3_BUCKET, fkey):
                return [media_url]

            upload_file_in_mem(
                self.__s3cli, r.content, S3_BUCKET, fkey
            )

            logger.info(f"processed image {tweet['id']} -> {fkey}")
            return [media_url]

    def _download_video(
        self, tweet: Dict[str, Any], media_entity: Dict[str, Any]
    ) -> List[str]:
        max_bitrate = 0
        max_url = ""
        for v in media_entity["video_info"]["variants"]:
            if v.get("bitrate", 0) > max_bitrate:
                max_bitrate = v.get("bitrate", 0)
                max_url = v.get("url", "")

        if not max_url:
            return []

        # https://video.twimg.com/ext_tw_video/1485662967387492354/pu/vid/960x544/0jVjzJr2sRgfSPzc.mp4?tag=12
        max_url = max_url[: max_url.rfind("?")]

        fkey = get_s3_key(media_entity, fname=max_url.split("/")[-1])
        media_url = f"https://s3.laisky.com/{S3_BUCKET}/{fkey}"
        with requests.get(max_url) as r:
            assert (
                r.status_code == 200
            ), f"download {max_url}: [{r.status_code}]{r.content}"

            if is_file_exists(self.__s3cli, len(r.content), S3_BUCKET, fkey):
                return [media_url]

            upload_file_in_mem(
                self.__s3cli, r.content, S3_BUCKET, fkey
            )

            logger.info(f"processed video {tweet['id']} -> {fkey}")

        return [media_url]

    def download_medias_for_tweet(self, tweet: Dict[str, Any]):
        media = tweet.get("extended_entities", {}).get("media", []) or tweet.get(
            "entities", {}
        ).get("media", [])

        # if self.is_download_media(tweet):
        #     return

        logger.info(f'download medias for tweet {tweet.get("id_str")}')
        medias = []
        for img in media:
            if img["type"] == "photo":
                medias += self._download_image(tweet, img)
            elif img["type"] == "video":
                medias += self._download_video(tweet, img)

        if not medias:
            return

        replace_media_urls(tweet, medias)

    def is_download_media(self, tweet: Dict[str, Any]) -> bool:
        """only download medias for user in users table, to save disk space"""
        user_id = tweet.get("user", {}).get("id_str", "").strip()
        if not user_id:
            return False

        return self.db["users"].find_one({"id_str": user_id}) is not None

    def g_load_user(self):
        logger.debug("g_load_user_auth")

        for u in self.db["account"].find():
            yield u

    def _save_relate_tweets(self, status: Dict[str, Any]):
        logger.debug(f"_save_relate_tweets for {status['id']}")
        try:
            for id_ in gen_related_tweets(self.col, status):
                try:
                    try:
                        docu = self.api.get_status(id_, tweet_mode="extended")
                    except tweepy.error.RateLimitError:
                        time.sleep(10)
                    except Exception:
                        logger.exception(f"load tweet {id_=} got error")
                        raise
                    else:
                        self.save_tweet(docu)
                        self._save_relate_tweets(docu)
                except Exception as err:
                    logger.exception(f"save tweet {id_=}")
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

    def crawl_timeline(self):
        """routine work to crawling twitter tweets"""
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
                    logger.exception(f"save user {u['username']} tweets")
        except Exception as err:
            logger.exception(err)
        else:
            logger.info("save {} tweets".format(count))

    def delete_old_tweets(self):


    def run_for_archive_data(self, user_id: int, tweet_fpath: str):
        self._current_user_id = user_id
        user = self.db["users"].find_one({"id_str": str(user_id)})
        assert user
        del user["_id"]

        with open(tweet_fpath) as fp:
            for tweet in json.loads(fp.read()):
                tweet = tweet["tweet"]
                tweet["user"] = user
                self._save_relate_tweets(tweet)
                self._save_replies(tweet)
                self.save_tweet(tweet)
