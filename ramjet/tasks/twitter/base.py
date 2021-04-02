import datetime
import random
import re
from typing import Dict, Generator, List

import pymongo
from ramjet.settings import logger as ramjet_logger

logger = ramjet_logger.getChild("tasks.twitter")
twitter_surl_regex = re.compile("https?://t\.co/[A-z0-9]*")


def get_tweet_text(tweet: Dict[str, any]) -> str:
    return tweet.get("full_text") or tweet.get("text", "")


def replace_to_laisky_url(url: str) -> str:
    srv_addr = random.choice(
        [
            "s1.laisky.com",
            "s2.laisky.com",
            "s3.laisky.com",
        ]
    )

    url = url.replace(
        "http://pbs.twimg.com/media/", f"https://{srv_addr}/uploads/twitter/"
    )
    url = url.replace(
        "https://pbs.twimg.com/media/", f"https://{srv_addr}/uploads/twitter/"
    )

    return url


def replace_media_urls(tweet: Dict[str, any]) -> None:
    for ee in ["entities", "extended_entities"]:
        for media in tweet.get(ee, {}).get("media", []):
            surl = media.get("url", "")
            durl = media.get("media_url_https") or media.get("media_url")
            durl = replace_to_laisky_url(durl)
            if durl:
                tweet["text"] = tweet["text"].replace(surl, durl)
                media["media_url_https"] = durl


def replace_short_urls(tweet: Dict[str, any]) -> None:
    for media in tweet.get("entities", {}).get("urls", []):
        surl = media.get("url", "")
        durl = media.get("expanded_url")
        if durl:
            tweet["text"] = tweet["text"].replace(surl, durl)


def twitter_api_parser(tweet: Dict[str, any]) -> Dict[str, any]:
    """Parse tweet document got from twitter api"""
    reg_topic = re.compile(r"[\b|\s]#(\S+)")
    tweet["topics"] = reg_topic.findall(get_tweet_text(tweet).replace(".", "_"))
    tweet["created_at"] = datetime.datetime.strptime(
        tweet["created_at"], "%a %b %d %H:%M:%S +0000 %Y"
    )

    tweet["id"] = int(tweet["id"])
    tweet["id_str"] = str(tweet["id"])

    # replace url
    tweet["text"] = get_tweet_text(tweet)
    replace_media_urls(tweet)
    replace_short_urls(tweet)

    return tweet


def gen_related_tweets(
    # tweetCol: pymongo.collection.Collection,
    tweet: Dict[str, any],
) -> List[str]:
    related_ids = []
    tweet.get("in_reply_to_status_id") and related_ids.append(
        tweet["in_reply_to_status_id"]
    )
    tweet.get("retweeted_status") and related_ids.append(
        tweet["retweeted_status"]["id"]
    )
    tweet.get("quoted_status") and related_ids.append(tweet["quoted_status"]["id"])
    return related_ids
    # for _id in filter(lambda id_: not tweetCol.find_one({"id": id_}), related_ids):
    #     yield _id


def twitter_history_parser(tweet: Dict[str, any]):
    """NOTIMPLEMENT!!!

    Tweets that from twitter history propose:
    ::
        {
            "id" : NumberLong("541582834209918976"),
            "text" : "digitalocean 延迟略高啊…等招行全币卡办下来我还是去买 linode 好了…",
            "in_reply_to_status_id" : null,
            "retweeted_status_timestamp" : null,
            "in_reply_to_user_id" : null,
            "source" : "<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>",
            "created_at" : ISODate("2014-12-07T13:19:44Z"),
            "retweeted_status_user_id" : null,
            "topics" : [ ],
            "expanded_urls" : null,
            "retweeted_status_id" : null
        }
    """
    raise NotImplementedError

    t = get_tweet_text(tweet)
    if "expanded_urls" not in tweet and "t.co" not in t:
        return tweet

    eurls = tweet["expanded_urls"].split(",")
    surls = twitter_surl_regex.findall(t)
    if len(eurls) != len(surls):
        _u = set()
        eurls = [u for u in eurls if u not in _u and not _u.add(u)]
        if len(eurls) != len(surls):
            err = "length of expanded_urls not equal to short_urls"
            logger.error("{} for tweet id {}".format(err, tweet["_id"]))

    for i, surl in enumerate(surls):
        ul = len(surl)
        ui = t.index(surl)
        if ul + ui != len(t):
            if t[ul + ui] != " ":
                t = t[: ui + ul] + " " + t[ui + ul :]

        if ui != 0:
            if t[ui - 1] != " ":
                t = t[:ui] + " " + t[ui:]

        t = t.replace(surl, eurls[i])

    tweet["text"] = t
    return tweet
