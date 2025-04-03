from concurrent.futures import wait, ALL_COMPLETED
from datetime import datetime
import bson.json_util
from random import choice
from typing import Dict
import hashlib
from dataclasses import dataclass


import pymongo
from kipp.decorator import timer
import pymongo.collection
import requests
from bs4 import BeautifulSoup

from ramjet.utils.log import logger as ramjet_logger
from ramjet.engines import thread_executor as executor
from ramjet.settings import prd
from .db import get_db

logger = ramjet_logger.getChild("tasks.telegram.notes")


def run():
    notes = get_notes_col()
    latest_post_id = get_latest_post_id(notes)

    futures = []
    for i in range(latest_post_id, latest_post_id + 5):
        f = executor.submit(fetch_content, notes, i)
        futures.append(f)

    wait(futures, return_when=ALL_COMPLETED)


def get_latest_post_id(notes: pymongo.collection.Collection) -> int:
    docu = notes.find_one(sort=[("post_id", pymongo.DESCENDING)])
    if not docu:
        return 0

    return docu["post_id"]


def get_notes_col() -> pymongo.collection.Collection:
    """get notes collection"""
    db = get_db()
    notes: pymongo.collection.Collection = db["notes"]

    # add index to db
    notes.create_index([("post_id", pymongo.ASCENDING)], unique=True)
    # notes.create_index([("content", pymongo.TEXT)]) # add text index for field `content`

    return notes


def upload_akord(filecontent: bytes) -> str:
    """upload file to akord

    Args:
        filecontent (bytes): file content

    Returns:
        str: txid
    """
    url = "https://api.akord.com/files"
    apikey = choice(prd.AKORD_APIKEYs)
    resp = requests.post(
        url,
        data=filecontent,
        headers={
            "Accept": "application/json",
            "Api-Key": apikey,
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    assert resp.status_code == 200, f"[{resp.status_code}]{resp.text}"

    return resp.json()["tx"]["id"]


@timer
def _upload_one_post(notes: pymongo.collection.Collection, docu: Dict):
    txid = upload_akord(bson.json_util.dumps(docu).encode("utf-8"))
    notes.update_one({"_id": docu["_id"]}, {"$set": {"akord_txid": txid}})
    logger.info(f"succeed uploaded {docu['post_id']=} to arweave")


@timer
def upload_all_posts(notes: pymongo.collection.Collection):
    # for docu in notes.find({"arweave_id": {"$exists": False}}):
    fs = []
    for docu in notes.find():
        fs.append(executor.submit(_upload_one_post, notes, docu))
        break

    wait(fs, return_when=ALL_COMPLETED)


@dataclass
class NoteHistory:
    """History record for a note"""
    content: str
    created_at: datetime
    digest: str


def calculate_sha256(content: str) -> str:
    """Calculate SHA256 hash of content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


@timer
def fetch_content(notes: pymongo.collection.Collection, post_id: int):
    """fetch content from telegram"""
    logger.debug(f"fetch_content {post_id=}")
    url = f"https://t.me/laiskynotes/{post_id}"
    resp = requests.get(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept-Language": "en,zh-CN;q=0.9,zh-TW;q=0.8,zh;q=0.7,fr;q=0.6",
        },
        timeout=30,
    )
    assert resp.status_code == 200

    # extract content
    soup = BeautifulSoup(resp.text, "html.parser")
    ele = soup.select_one("head > meta:nth-child(8)")
    if not ele:
        logger.info(f"cannot find element in {post_id=}")
        return

    content = ele.attrs["content"]
    if (
        not content.strip()
        or "记录和分享有趣的信息。 		Record and share interesting information." in content
    ):
        logger.info(f"cannot find content in {post_id=}")
        return

    # extract image
    #     image_urls = []
    #     images = soup.select("body > div > div.tgme_widget_message_bubble > div.tgme_widget_message_grouped_wrap.js-message_grouped_wrap > div > div a[style]")
    #     for img in images:
    #         image_urls.append(img.attrs["style"])
    # Calculate digest for new content
    new_digest = calculate_sha256(content)

    # Get existing note
    existing_note = notes.find_one({"post_id": post_id})
    now = datetime.now()

    if existing_note:
        # Calculate digest for existing content if not present
        existing_digest = existing_note.get("digest")
        if not existing_digest:
            existing_digest = calculate_sha256(existing_note["content"])

        # Compare digests
        if existing_digest == new_digest:
            logger.info(f"Content unchanged for {post_id=}")
            return

        # Create history entry
        history_entry = {
            "content": existing_note["content"],
            "created_at": existing_note.get("updated_at", existing_note["created_at"]),
            "digest": existing_digest,
        }

        # Update with new content and add to history
        notes.update_one(
            {"post_id": post_id},
            {
                "$set": {"content": content, "updated_at": now, "digest": new_digest},
                "$push": {
                    "history": {
                        "$each": [history_entry],
                        "$position": 0
                    }
                },
            },
        )
    else:
        # Insert new document
        notes.update_one(
            {"post_id": post_id},
            {
                "$set": {
                    "post_id": post_id,
                    "content": content,
                    "updated_at": now,
                    "digest": new_digest,
                    "history": [],
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

    logger.info(f"Updated content for {post_id=}")
