import os
import re
from urllib.parse import urlsplit
from base64 import b64decode
from io import BytesIO

import openai
import httpx
from minio import Minio

from ramjet.settings import prd
from ramjet.tasks.gptchat.utils import logger
from ramjet.tasks.gptchat.credentials import (
    DEFAULT_API_BASE,
    resolve_model_credentials,
    resolve_sdk_credentials,
)

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


def resolve_image_parameters(
    api_base: str,
    model: str | None = None,
    image_profile: str | None = None,
) -> dict:
    """resolve_image_parameters returns explicit model-specific generation options.

    Only the standard OpenAI HTTPS API receives a default model. Custom providers
    require a model; unknown model names also require a declared parameter profile.
    No endpoint, credential or alternative model is substituted.
    """
    parsed = urlsplit(api_base)
    standard = (
        parsed.scheme == "https"
        and parsed.hostname == "api.openai.com"
        and parsed.port in {None, 443}
        and parsed.username is None
        and parsed.password is None
        and parsed.path.rstrip("/") == "/v1"
        and not parsed.query
        and not parsed.fragment
    )
    if model is None:
        if not standard:
            raise ValueError("Custom image providers require an explicit model")
        model = "gpt-image-2-2026-04-21"
    if not isinstance(model, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model
    ):
        raise ValueError("A valid image model is required")
    legacy = model in {"dall-e-2", "dall-e-3"}
    gpt_image = model in {
        "gpt-image-1",
        "gpt-image-1-mini",
        "gpt-image-1.5",
        "gpt-image-2",
        "gpt-image-2-2026-04-21",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-2026-09-08",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-2026-09-08",
        "chatgpt-image-latest",
    }
    if standard and legacy:
        raise ValueError("Retired DALL-E models are unsupported on OpenAI")
    if image_profile is None:
        if legacy:
            image_profile = "legacy"
        elif gpt_image:
            image_profile = "gpt-image"
        else:
            raise ValueError("Unknown image models require an explicit image_profile")
    if not isinstance(image_profile, str) or image_profile not in {
        "gpt-image",
        "legacy",
    }:
        raise ValueError("image_profile must be gpt-image or legacy")
    if (legacy and image_profile != "legacy") or (
        gpt_image and image_profile != "gpt-image"
    ):
        raise ValueError("image_profile conflicts with the selected model")
    if standard and (not gpt_image or image_profile != "gpt-image"):
        raise ValueError("OpenAI image models require the GPT Image profile")
    parameters = {"model": model, "n": 1, "size": "1024x1024"}
    if image_profile == "gpt-image":
        parameters.update(output_format="png", quality="low")
    else:
        parameters["response_format"] = "b64_json"
    return parameters


def draw_image_by_dalle(
    prompt: str,
    apikey: str,
    api_base: str = DEFAULT_API_BASE,
    model: str | None = None,
    image_profile: str | None = None,
) -> bytes:
    """draw_image_by_dalle returns image bytes from the caller's selected provider.

    The shared resolver binds the explicit key/backend. Model capabilities select
    compatible parameters before SDK construction; public image bytes are retained.
    """
    credentials = resolve_model_credentials(apikey, api_base)
    parameters = resolve_image_parameters(credentials["base_url"], model, image_profile)
    options = resolve_sdk_credentials(apikey, api_base, include_async_client=False)
    with openai.OpenAI(**options) as client:
        response = client.images.generate(prompt=prompt, **parameters)

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
    prompt: str,
    apikey: str,
    api_base: str | None = None,
    model: str | None = None,
    image_profile: str | None = None,
) -> bytes:
    """draw_image_by_dalle_azure retains the legacy name with explicit options.

    This helper never invents native Azure configuration, a model or a server key.
    Its backend must expose the selected compatible image-generation operation.
    """
    if api_base is None:
        raise ValueError("An explicit image provider URL is required")
    return draw_image_by_dalle(
        prompt=prompt,
        apikey=apikey,
        api_base=api_base,
        model=model,
        image_profile=image_profile,
    )
