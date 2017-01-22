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
from aiohttp_session import get_session
import aiohttp_jinja2

from .base import logger
from ramjet.utils import get_conn, utcnow, obj2str, str2obj, generate_token
from ramjet.settings import CONSUMER_KEY, CONSUMER_SECRET


def bind_handle(add_route):
    logger.info('bind_handle')
    add_route('login/', LoginHandle)
    add_route('oauth/', OAuthHandle)


def get_auth():
    return tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)


class LoginHandle(web.View):

    async def get(self):
        logger.info('GET LoginHandle')

        s = await get_session(self.request)
        auth = get_auth()
        url = auth.get_authorization_url()
        resp = web.HTTPFound(url)
        s_token = obj2str(auth.request_token)
        s['request_token'] = s_token
        return resp


class OAuthHandle(web.View):

    @aiohttp_jinja2.template('twitter/login.html')
    async def get(self):
        """
        OAuth 登陆的回调地址
            https://laisky.com/ramjet/twitter/oauth/
        """
        logger.info('GET OAuthHandle')

        session = await get_session(self.request)
        if not session or not session.get('request_token'):
            return web.Response(text="Please enable cookies!")

        req_token = str2obj(session.get('request_token'))
        auth = get_auth()
        ql = urllib.parse.parse_qs(self.request.query_string)
        try:
            verify = ql['oauth_verifier'][0]
        except Exception:
            return web.Response(text="OAuth Error")

        auth.request_token = req_token
        access_token, access_token_secret = auth.get_access_token(verify)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

        docu = self.get_userinfo()
        docu.update({'access_token_secret': access_token_secret,
                     'access_token': access_token,
                     'last_update': utcnow()})
        self.save_userinfo(docu)
        session['username'] = docu['username']

        docu = {'id': 123, 'username': 'laisky'}
        token = {
            'source': 'twitter',
            'id': docu['id'],
            'username': docu['username'],
        }

        return {
            'info': 'welcome {}'.format(docu['username']),
            'token': generate_token(token),
        }

    def save_userinfo(self, docu):
        logger.info('save_userinfo for {}'.format(docu['username']))

        conn = get_conn()
        col = conn['twitter']['account']
        col.update(
            {'id': docu['id']},
            {'$set': docu},
            upsert=True
        )

    def get_userinfo(self):
        logger.debug('get_userinfo')

        docu = self.api.verify_credentials()  # return object
        return {
            'id': docu.id,
            'username': docu.name,
        }
