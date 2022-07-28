from .base import logger, parse_tweet_text
from .crawler import TwitterAPI


def main():
    api = TwitterAPI()
    # replace_pbs_img_url(api)
    replace_tco_short_url(api)


def replace_tco_short_url(api: TwitterAPI):
    total = processed = 0
    for tweet in api.col.find(
        # {"full_text": {"$regex": r"/https://t\.co//"}},
        {},
        no_cursor_timeout=True,
    ):
        if total % 100 == 0:
            logger.info(f"processed {processed}/{total}")

        total += 1
        if not tweet.get("text"):
            continue

        text = tweet["text"]
        if "https://t.co/" not in text:
            continue

        tweet = parse_tweet_text(tweet)
        api.download_medias_for_tweet(tweet)

        print(f">> {tweet['id_str']} : {tweet['text']}, {tweet.get('full_text')}")
        # continue

        api.col.update_one({"_id": tweet["_id"]}, {"$set": tweet})
        processed += 1


def replace_pbs_img_url(api: TwitterAPI):
    """remove twitter pbs url in tweets

    ::
        https://pbs.twimg.com/ext_tw_video_thumb/1550911725381222402/pu/img/lY5GUM_9pTdvV29O.jpg
    """

    total = processed = 0
    for tweet in api.col.find(
        {"text": {"$regex": r"/pbs\.twimg\.com/"}}, no_cursor_timeout=True
    ):
        if total % 100 == 0:
            logger.info(f"processed {processed}/{total}")

        total += 1
        if not tweet.get("text"):
            continue

        text = tweet["text"]
        if "pbs.twimg.com" not in text:
            continue

        try:
            api.download_medias_for_tweet(tweet)
        except Exception:
            logger.exception(f"download medias for tweet {tweet['id']}")
            continue

        print(f">> {tweet['id_str']} : {tweet['text']}")
        # return

        api.col.update_one({"_id": tweet["_id"]}, {"$set": {"text": tweet["text"]}})
        processed += 1


if __name__ == "__main__":
    main()
