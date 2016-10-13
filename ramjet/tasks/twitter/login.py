"""
auth = tweepy.OAuthHandler("consumer_key", "consumer_secret")

# Redirect user to Twitter to authorize
redirect_user(auth.get_authorization_url())

    http://hime.laisky.us/api/authorize?source=twitter
        &oauth_token=xxxx
        &oauth_verifier=yyyy

# Get access token
access_token, access_token_secret = auth.get_access_token("verifier_value")
# auth.access_token
# auth.access_token_secret
"""

import urllib

from aiohttp import web
import tweepy

from .base import logger
from ramjet.utils import get_conn, utcnow
from ramjet.settings import CONSUMER_KEY, CONSUMER_SECRET


def bind_handle(app):
    logger.info('bind_handle')
    app.router.add_route('*', '/twitter/login/', LoginHandle)
    app.router.add_route('*', '/twitter/oauth/', OAuthHandle)


def get_auth():
    return tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)


class LoginHandle(web.View):

    async def get(self):
        auth = get_auth()
        url = auth.get_authorization_url()
        return web.HTTPFound(url)


class OAuthHandle(web.View):

    async def get(self):
        """
        OAuth 登陆的回调地址
            https://laisky.com/ramjet/twitter/oauth/
        """
        auth = get_auth()
        ql = urllib.parse.parse_qs(self.request.query_string)
        verify = ql['oauth_verifier'][0]
        access_token, access_token_secret = auth.get_access_token(verify)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

        docu = self.get_userinfo()
        docu.update({'access_token_secret': access_token_secret,
                     'access_token': access_token,
                     'last_update': utcnow()})
        self.save_userinfo(docu)
        return web.Response(text="login ok")

    def save_userinfo(self, docu):
        conn = get_conn()
        col = conn['twitter']['account']
        col.update(
            {'id': docu['id']},
            {'$set': docu},
            upsert=True
        )

    def get_userinfo(self):
        docu = api.verify_credentials()
        return {
            'id': docu['id'],
            'username': docu['name'],
        }
