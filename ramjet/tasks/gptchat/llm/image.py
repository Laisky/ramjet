import os
from base64 import b64decode
from io import BytesIO

import openai
from minio import Minio

from ramjet.settings import prd
from ramjet.tasks.gptchat.utils import logger

logger = logger.getChild("image")


def image_objkey(task_id: str) -> str:
    """get image url path without schema and domain by task id

    Args:
        task_id (str): task id

    Returns:
        str: image url path
    """
    return f"{prd.OPENAI_S3_CHUNK_CACHE_IMAGES}/{task_id[:2]}/{task_id[2:4]}/{task_id}.png"


def upload_image_to_s3(
    s3cli: Minio, task_id: str, prompt: str, img_content: bytes
) -> str:
    """upload image to s3

    Args:
        s3cli (Minio): s3 client
        img_content (bytes): image content
        task_id (str): task id
        prompt (str): prompt be used to generate the image

    Returns:
        str: image url
    """
    objkey_prefix = os.path.splitext(image_objkey(task_id=task_id))[0]
    logger.debug(f"wait upload image and prompt to s3, key={objkey_prefix}")

    # upload image
    s3cli.put_object(
        bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
        object_name=f"{objkey_prefix}.png",
        data=BytesIO(img_content),
        length=len(img_content),
    )

    # upload prompt
    s3cli.put_object(
        bucket_name=prd.OPENAI_S3_CHUNK_CACHE_BUCKET,
        object_name=f"{objkey_prefix}.txt",
        data=BytesIO(prompt.encode("utf-8")),
        length=len(prompt.encode("utf-8")),
    )

    logger.info(f"succceed upload image and prompt to s3, key={objkey_prefix}")
    return f"{prd.S3_SERVER}/{prd.OPENAI_S3_CHUNK_CACHE_BUCKET}/{objkey_prefix}.png"


def draw_image_by_dalle(prompt: str, apikey: str) -> bytes:
    """generate image from prompt by openai dalle

    ref: https://platform.openai.com/docs/api-reference/images/create

    Args:
        prompt (str): description to draw the image
        apikey (str): openai api key

    Returns:
        bytes: the image in bytes, can be save as png file
    """
    response: dict = openai.Image.create(
        api_base="https://api.openai.com/v1/",  # only openai support dalle
        prompt=prompt,
        api_key=apikey,
        n=1,
        # size="1024x1024",
        size="512x512",
        response_format="b64_json",
    )

    logger.debug(f"succeed draw image by dalle-2, {prompt=}")
    return b64decode(response["data"][0]["b64_json"])
