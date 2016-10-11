
import pymongo
from tweepy import API, OAuthHandler

from ramjet.engines import ioloop, thread_executor
from ramjet.settings import CONSUMER_KEY, CONSUMER_SECRET, \
    ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from ramjet.utils import db_conn
from .base import twitter_api_parser, logger


def bind_task():
    def run():
        logger.info('run')
        twitter = TwitterAPI()
        thread_executor.submit(twitter.run)

        later = 3600
        ioloop.call_later(later, run)

    run()


class TwitterAPI:

    __api = None

    @property
    def api(self):
        logger.debug('get api')
        if self.__api:
            return self.__api

        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.__api = API(auth)
        return self.__api

    def g_load_tweets(self, since_id):
        logger.debug('g_load_tweets for since_id {}'.format(since_id))
        for s in self.api.user_timeline(since_id=since_id):
            yield s

    @property
    def col(self):
        return db_conn['twitter']['tweets']

    def parse_tweet(self, tweet):
        logger.debug('parse_tweet')
        return twitter_api_parser(tweet._json)

    def get_last_tweet_id(self):
        logger.debug('get_last_tweet_id')
        docu = self.col.find_one(sort=[('id', pymongo.DESCENDING)])
        return docu['id']

    def save_tweet(self, docu):
        logger.debug('save_tweet')
        self.col.update(
            {'id': docu['id']},
            {'$set': docu},
            upsert=True
        )

    def run(self):
        logger.debug('run TwitterAPI')
        try:
            last_id = self.get_last_tweet_id()
            for count, status in enumerate(self.g_load_tweets(last_id)):
                tweet = self.parse_tweet(status)
                self.save_tweet(tweet)
        except Exception as err:
            logger.exception(err)
        else:
            logger.info('save {} tweets'.format(count))
