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

import urllib.parse
from typing import Dict

import aiohttp_jinja2
import tweepy
from aiohttp import web
from aiohttp_session import get_session
from ramjet.settings import (
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_CLIENT_ID,
    TWITTER_CALLBACK_URL,
    TWITTER_CLIENT_SECRET,
)
from ramjet.utils import generate_token, get_conn, obj2str, str2obj, utcnow, recover

from .base import logger


def get_auth():
    """get twitter api v1.1 auth handler"""
    return tweepy.OAuth1UserHandler(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        callback=TWITTER_CALLBACK_URL,
    )


def get_auth_v2() -> tweepy.OAuth2UserHandler:
    """get twitter api v2 auth handler

    only work for developer account,
    if you want let normal user login, use `get_auth` instead
    """
    return tweepy.OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_CALLBACK_URL,
        scope=["tweet.read", "users.read", "follows.read", "like.read"],
        # Client Secret is only necessary if using a confidential client
        client_secret=TWITTER_CLIENT_SECRET,
    )


class LoginHandle(web.View):
    # @recover
    # async def get(self):
    #     """generate api login url for API v1"""
    #     logger.info("GET LoginHandle")

    #     s = await get_session(self.request)
    #     auth = get_auth()
    #     url = auth.get_authorization_url()
    #     resp = web.HTTPFound(url)
    #     s_token = obj2str(auth.request_token)
    #     s["request_token"] = s_token

    #     logger.debug("someone try to login via twitter")
    #     return resp

    @recover
    async def get(self):
        """generate api login url for API v2

        Refs:
            - https://docs.tweepy.org/en/stable/authentication.html
        """
        logger.info("GET LoginHandle")

        oauth = get_auth()
        login_url = oauth.get_authorization_url(signin_with_twitter=True)

        resp = web.HTTPFound(login_url)
        logger.debug("someone try to login via twitter API v2")
        return resp


class OAuthHandle(web.View):
    # @recover
    # @aiohttp_jinja2.template("twitter/login.html")
    # async def get(self):
    #     """
    #     twitter api v1.1 oauth login

    #     OAuth 登陆的回调地址
    #         https://app.laisky.com/twitter/oauth/
    #     """
    #     logger.info("GET OAuthHandle")

    #     session = await get_session(self.request)
    #     if not session or not session.get("request_token"):
    #         return web.Response(text="Please enable cookies!")

    #     req_token = str2obj(session.get("request_token"))
    #     auth = get_auth()
    #     ql = urllib.parse.parse_qs(self.request.query_string)
    #     try:
    #         verify = ql["oauth_verifier"][0]
    #     except Exception:
    #         return web.Response(text="OAuth Error")

    #     auth.request_token = req_token
    #     access_token, access_token_secret = auth.get_access_token(verify)
    #     auth.set_access_token(access_token, access_token_secret)
    #     self.api = tweepy.API(auth)

    #     docu = self.get_userinfo()
    #     docu.update(
    #         {
    #             "access_token_secret": access_token_secret,
    #             "access_token": access_token,
    #             "last_update": utcnow(),
    #         }
    #     )
    #     self.save_userinfo(docu)
    #     session["username"] = docu["username"]

    #     docu = {"id": 123, "username": "laisky"}
    #     token = {
    #         "source": "twitter",
    #         "id": docu["id"],
    #         "username": docu["username"],
    #     }

    #     logger.info(f"succeed login from twitter, user={docu['username']}")
    #     return {
    #         "info": "welcome {}".format(docu["username"]),
    #         "token": generate_token(token),
    #     }

    @recover
    @aiohttp_jinja2.template("twitter/login.html")
    async def get(self):
        """
        twitter api v2 oauth login

        OAuth 登陆的回调地址
            https://app.laisky.com/twitter/oauth/

        Example:
            https://app.laisky.com/twitter/oauth/?oauth_token=bJhWqAAAAAABoxCvAAABiU3wxXU&oauth_verifier=nzXXwTwTiWKpdLtCmO8QYqcKryxVQ54w
        """
        logger.info("GET OAuthHandle")

        # auth = twitter_api_v2_auth()
        # bearer_token = auth.fetch_token(self.request.url)
        # client = tweepy.Client(bearer_token=bearer_token)

        ql = urllib.parse.parse_qs(self.request.query_string)
        verifier = ql["oauth_verifier"][0]

        auth  = get_auth()
        access_token, access_token_secret = auth.get_access_token(verifier=verifier)
        client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )

        # Returns:
        # {
        #   "data": {
        #     "id": "2244994945",
        #     "name": "TwitterDev",
        #     "username": "Twitter Dev"
        #   }
        # }
        resp = client.get_me()
        assert isinstance(resp, Dict), f"get_me got: {resp}"

        user_info = resp["data"]
        user_info.update(
            {
                "access_token": access_token,
                "access_token_secret": access_token_secret,
                "last_update": utcnow(),
            }
        )
        self.save_userinfo(user_info)

        token = {
            "source": "twitter",
            "id": user_info["id"],
            "username": user_info["username"],
        }

        logger.info(f"succeed login from twitter, user={user_info['username']}")
        return {
            "info": "welcome {}".format(user_info["username"]),
            "token": generate_token(token),
        }

    def save_userinfo(self, docu):
        logger.info("save_userinfo for {}".format(docu["username"]))

        conn = get_conn()
        col = conn["twitter"]["account"]
        col.update_one({"id": docu["id"]}, {"$set": docu}, upsert=True)
