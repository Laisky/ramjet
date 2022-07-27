from .crawler import TwitterAPI

if __name__ == "__main__":
    t = TwitterAPI()
    # t.run_for_archive_data(
    #     user_id=105351466,
    #     # r"/mnt/c/Users/ppcel/Downloads/twitter-2021-01-30-171d328c51ae57206ad8ceb533c4e435c9a6b4ca7ec37f3441d01ab1d4f64cd2/data/tweet.js/"
    #     tweet_fpath=r"/home/laisky/tweet.js"
    # )

    t.run_for_replace_all_twitter_url()
