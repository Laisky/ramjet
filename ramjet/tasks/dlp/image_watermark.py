import json
import os
import tempfile
from base64 import b64encode
from hashlib import md5
from uuid import uuid1

import aiohttp_jinja2
import requests
from aiohttp import web
from ramjet.settings import (TENCENT_CLOUD_SIGNATURE,
                             TENCENT_CLOUD_WATERMARK_BUCKET_URL, logger)


class ImageWaterMarkView(web.View):
    async def get(self):
        """get watermarked image

        `/dlp/image/watermark/?url=xxx&text=xxx`
        """
        img_remote_url = self.request.rel_url.query["url"]
        watermark = b64encode(self.request.rel_url.query["text"].encode("utf8"))
        hasher = md5()
        hasher.update(img_remote_url.encode("utf8"))
        key = hasher.hexdigest()

        img_url = tencent_cloud.fetch_and_upload_image(img_remote_url, key)
        redirect_to = f"{img_url}?watermark/3/type/3/text/{watermark}"
        raise web.HTTPFound(redirect_to)

class ImageWaterMarkView(web.View):
    @aiohttp_jinja2.template("dlp/image_watermark_doc.html")
    async def get(self):
        return


class ImageWaterMarkVerifyView(web.View):
    @aiohttp_jinja2.template("dlp/image_watermark_verify.html")
    async def get(self):
        """verify watermarked image

        `/dlp/image/watermark/verify`
        """
        return

    async def post(self):
        data = await self.request.post()
        print(data.keys())
        watermark_url=tencent_cloud.load_watermarked_image_url(data['image'])
        raise web.HTTPFound(watermark_url)


class TencentCloud:
    _client = None

    def __init__(self):
        self._client = requests.Session()
        self._client.headers.update(
            {
                "Authorization": TENCENT_CLOUD_SIGNATURE,
            }
        )

    def fetch_and_upload_image(self, img_url: str, key: str) -> str:
        resp = self._client.get(img_url)
        assert resp.status_code == 200, f"got resp [{resp.status_code}]: {resp.text}"

        logger.info(f"downloaded image {img_url=}")
        with tempfile.TemporaryFile() as fp:
            fp.write(resp.content)
            fp.seek(0, 0)

            resp = self._client.put(
                url=f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}/{key}", data=fp.read()
            )
            assert (
                resp.status_code == 200
            ), f"got resp [{resp.status_code}]: {resp.text}"

            logger.info(f"uploaded image {img_url=}")
            return f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}/{key}"

    def load_watermarked_image_url(self, imgfile) -> str:
        # upload image to cloud
        fname = f"{uuid1()}.{os.path.splitext(imgfile.filename)[1]}"
        upload_key = f"/verify/{fname}"
        extract_key = f"/extracted/{fname}"
        resp = self._client.put(
            url=f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}{upload_key}",
            data=imgfile.file.read(),
            headers={
                "Pic-Operations": json.dumps(
                        {
                            "is_pic_info": 0,
                            "rules": [
                                {
                                    "fileid": extract_key,
                                    "rule": "watermark/4/type/3/text/xxx",
                                }
                            ],
                        }
                )
            },
        )
        assert resp.status_code == 200, f"got resp [{resp.status_code}]: {resp.text}"

        logger.info(f"uploaded verify image {fname=}")
        return f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}{extract_key}"


tencent_cloud = TencentCloud()
