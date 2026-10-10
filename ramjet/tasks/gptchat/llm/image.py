import os
from base64 import b64decode
from io import BytesIO

import openai
import httpx
from minio import Minio

from ramjet.settings import prd
from ramjet.tasks.gptchat.utils import logger
from ramjet.tasks.gptchat.credentials import DEFAULT_API_BASE, resolve_sdk_credentials

logger = logger.getChild("image")


def image_objkey(task_id: str) -> str:
    """get image url path without schema and domain by task id

    Args:
        task_id (str): task id

    Returns:
        str: image url path
    """
    return (
        f"{prd.OPENAI_S3_CHUNK_CACHE_IMAGES}/{task_id[:2]}/{task_id[2:4]}/{task_id}.png"
    )


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


def draw_image_by_dalle(
    prompt: str, apikey: str, api_base: str = DEFAULT_API_BASE
) -> bytes:
    """draw_image_by_dalle returns image bytes from the caller's selected provider.

    The installed SDK receives an explicit key and backend through the shared
    resolver. DALL-E2 parameters and the public image-byte contract are retained.
    """
    options = resolve_sdk_credentials(apikey, api_base, include_async_client=False)
    with openai.OpenAI(**options) as client:
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )

    if not response.data:
        raise ValueError("The model provider returned no image")
    image = response.data[0]
    if image.b64_json:
        content = b64decode(image.b64_json)
    elif image.url:
        # Keep URL-only provider responses without forwarding the model key.
        downloaded = httpx.get(image.url, timeout=30, follow_redirects=True)
        downloaded.raise_for_status()
        content = downloaded.content
    else:
        raise ValueError("The model provider returned no image")
    logger.debug("Image generation completed")
    return content


def draw_image_by_dalle_azure(
    prompt: str, apikey: str, api_base: str | None = None
) -> bytes:
    """draw_image_by_dalle_azure retains the legacy name with an explicit backend.

    This compatibility helper never invents Azure configuration or a server key.
    Its backend must expose the same compatible image-generation operation.
    """
    if api_base is None:
        raise ValueError("An explicit image provider URL is required")
    return draw_image_by_dalle(prompt=prompt, apikey=apikey, api_base=api_base)
