
import pymongo
from tweepy import API, OAuthHandler

from ramjet.engines import ioloop, thread_executor
from ramjet.settings import CONSUMER_KEY, CONSUMER_SECRET, \
    ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from ramjet.utils import get_conn
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
    __auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

    @property
    def api(self):
        logger.debug('get api')
        if self.__api:
            return self.__api

        return self.set_api()

    def set_api(self, access_token=ACCESS_TOKEN, access_token_secret=ACCESS_TOKEN_SECRET):
        self.__auth.set_access_token(access_token, access_token_secret)
        self.__api = API(self.__auth)
        return self.__api

    def g_load_tweets(self, last_id):
        """
        Twitter 只能反向回溯，从最近的推文开始向前查找
        """
        logger.debug('g_load_tweets for last_id {}'.format(last_id))
        last_tweets = self.api.user_timeline(count=1)
        if not last_tweets:
            return

        yield last_tweets[0]
        current_id = last_tweets[0].id
        while True:
            tweets = self.api.user_timeline(max_id=current_id, count=100)
            if len(tweets) == 1:  # 到头了
                return

            for s in tweets:
                if s.id <= last_id:  # 已存储
                    return

                yield s
                if s.id < current_id:
                    current_id = s.id

    @property
    def col(self):
        return get_conn()['twitter']['tweets']

    @property
    def db(self):
        return get_conn()['twitter']

    def parse_tweet(self, tweet):
        logger.debug('parse_tweet')
        return twitter_api_parser(tweet._json)

    def get_last_tweet_id(self):
        """
        获取数据库里存储的最后一条推文
        """
        logger.debug('get_last_tweet_id')
        docu = self.db['tweets'].find_one(
            {'user.id': self.current_user_id},
            sort=[('id', pymongo.DESCENDING)]
        )
        return docu and docu['id']

    def save_tweet(self, docu):
        logger.debug('save_tweet')
        self.db['tweets'].update(
            {'id': docu['id']},
            {'$set': docu},
            upsert=True
        )

    def g_load_user(self):
        logger.debug('g_load_user_auth')

        for u in self.db['account'].find():
            yield u

    def run(self):
        logger.debug('run TwitterAPI')
        try:
            for u in self.g_load_user():
                self.current_user_id = u['id']
                self.set_api(u['access_token'], u['access_token_secret'])
                last_id = self.get_last_tweet_id() or 1
                for count, status in enumerate(self.g_load_tweets(last_id)):
                    tweet = self.parse_tweet(status)
                    self.save_tweet(tweet)
        except Exception as err:
            logger.exception(err)
        else:
            logger.info('save {} tweets'.format(count))
