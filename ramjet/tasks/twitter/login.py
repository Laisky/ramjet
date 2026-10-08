"""
auth = tweepy.OAuthHandler("consumer_key", "consumer_secret")

# Redirect user to Twitter to authorize
redirect_user(auth.get_authorization_url())

   https://app.laisky.com/twitter/oauth/?oauth_token=xxxx&oauth_verifier=yyyy

# Get access token
access_token, access_token_secret = auth.get_access_token("verifier_value")
# auth.access_token
# auth.access_token_secret
"""

import aiohttp_jinja2
import tweepy
from aiohttp import web
from aiohttp_session import get_session
from ramjet.settings import TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET
from ramjet.utils import generate_token, get_conn, utcnow
from ramjet.oauth_state import consume_request_token, make_request_token_state

from .base import logger


def get_auth():
    return tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)


class LoginHandle(web.View):
    async def get(self):
        logger.info("GET LoginHandle")

        s = await get_session(self.request)
        auth = get_auth()
        url = auth.get_authorization_url()
        resp = web.HTTPFound(url)
        s["request_token"] = make_request_token_state(auth.request_token)
        return resp


class OAuthHandle(web.View):
    @aiohttp_jinja2.template("twitter/login.html")
    async def get(self):
        """
        OAuth 登陆的回调地址
            https://app.laisky.com/twitter/oauth/
        """
        logger.info("GET OAuthHandle")

        session = await get_session(self.request)
        try:
            req_token, verify = consume_request_token(
                session, self.request.query_string
            )
        except (ValueError, TypeError):
            return web.Response(text="OAuth Error")

        auth = get_auth()
        auth.request_token = req_token
        try:
            access_token, access_token_secret = auth.get_access_token(verify)
        except tweepy.TweepyException:
            logger.warning("OAuth provider exchange failed")
            return web.Response(text="OAuth Error")
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

        docu = self.get_userinfo()
        docu.update(
            {
                "access_token_secret": access_token_secret,
                "access_token": access_token,
                "last_update": utcnow(),
            }
        )
        self.save_userinfo(docu)
        session["username"] = docu["username"]

        docu = {"id": 123, "username": "laisky"}
        token = {
            "source": "twitter",
            "id": docu["id"],
            "username": docu["username"],
        }

        return {
            "info": "welcome {}".format(docu["username"]),
            "token": generate_token(token),
        }

    def save_userinfo(self, docu):
        logger.info("save_userinfo for {}".format(docu["username"]))

        conn = get_conn()
        col = conn["twitter"]["account"]
        col.update({"id": docu["id"]}, {"$set": docu}, upsert=True)

    def get_userinfo(self):
        logger.debug("get_userinfo")

        docu = self.api.verify_credentials()  # return object
        return {
            "id": docu.id,
            "username": docu.name,
        }
