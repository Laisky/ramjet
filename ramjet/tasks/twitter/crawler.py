from pathlib import Path
from threading import RLock
from typing import Dict

import pymongo
import requests
import tweepy
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
        thread_executor.submit(twitter.run())

    run()


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

    def g_load_tweets(self, last_id: int):
        """
        Twitter 只能反向回溯，从最近的推文开始向前查找
        """
        logger.info("g_load_tweets for last_id {}".format(last_id))
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
        logger.debug("save_tweet")
        docu = self.parse_tweet(tweet)
        logger.info(f"save tweet {docu['id']}")
        user = docu.get("user")
        self.db["tweets"].update_one({"id": docu["id"]}, {"$set": docu}, upsert=True)
        if user:
            self.db["users"].update_one({"id": user["id"]}, {"$set": user}, upsert=True)

        self.download_images_for_tweet(tweet)

        self.db["tweets"].update_one(
            {"id": docu["id"]}, {"$addToSet": {"viewer": self._current_user_id}}
        )

    def download_images_for_tweet(self, tweet: Dict[str, any]):
        if not tweet.get("entities", {}).get("media"):
            return

        for img in tweet["entities"]["media"]:
            with requests.get(img["media_url_https"] + ":orig") as r:
                if r.status_code != 200:
                    logger.error(f"download error: [{r.status_code}]{r.content}")
                    continue

                fpath = Path(TWITTER_IMAGE_DIR, img["media_url_https"].split("/")[-1])
                if fpath.is_file():
                    continue

                with open(fpath, "wb") as f:
                    f.write(r.content)

                logger.info("tweet img ok", tweet["id"], fpath)

    def g_load_user(self):
        logger.debug("g_load_user_auth")

        for u in self.db["account"].find():
            yield u

    def _save_relate_tweets(self, status: Dict[str, any]):
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
                docu = self.api.get_status(id_)
            except Exception:
                logger.exception(f"load tweet {id_} got error")
            else:
                logger.info(f"save tweet [{docu['user']['screen_name']}]{docu['id']}")
                self.save_tweet(docu)
                self._save_relate_tweets(docu)

    def run(self):
        if not lock.acquire(blocking=False):
            return

        logger.debug("run TwitterAPI")
        count = 0
        try:
            for u in self.g_load_user():
                self._current_user_id = u["id"]
                self.set_api(u["access_token"], u["access_token_secret"])
                last_id = self.get_last_tweet_id() or 1
                for count, status in enumerate(self.g_load_tweets(last_id)):
                    self._save_relate_tweets(status)
                    self.save_tweet(status)
        except Exception as err:
            logger.exception(err)
        else:
            logger.info("save {} tweets".format(count))
        finally:
            lock.release()
