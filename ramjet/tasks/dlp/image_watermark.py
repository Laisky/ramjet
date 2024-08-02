import json
import os
import tempfile
from asyncio import wait
from base64 import urlsafe_b64encode
from hashlib import md5
from uuid import uuid1

import aiohttp
import aiohttp_jinja2
import requests
from aiohttp import web

from ramjet.utils.log import logger as ramjet_logger
from ramjet.settings import TENCENT_CLOUD_SIGNATURE, TENCENT_CLOUD_WATERMARK_BUCKET_URL


logger = ramjet_logger.getChild("dlp.image_watermark")


class ImageWaterMarkSignView(web.View):
    async def get(self):
        """get watermarked image

        `/dlp/image/watermark/?url=xxx&text=xxx`
        """
        img_remote_url = self.request.rel_url.query["url"]
        watermark = self.request.rel_url.query["text"]

        hasher = md5()
        hasher.update(img_remote_url.encode("utf-8"))
        key = hasher.hexdigest()

        imgdata = await tencent_cloud.fetch_and_sign_image(
            img_remote_url, key, watermark
        )
        return web.Response(headers={"content-type": "image/jpeg"}, body=imgdata)


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
        imgdata = await tencent_cloud.extract_watermark(data["image"])
        return web.Response(headers={"content-type": "image/jpeg"}, body=imgdata)


class TencentCloud:
    _client = None

    def __init__(self):
        self._client = aiohttp.ClientSession()
        self._client.headers.update(
            {
                "Authorization": TENCENT_CLOUD_SIGNATURE,
            }
        )

    async def fetch_and_sign_image(
        self, img_url: str, key: str, watermark: str
    ) -> bytearray:
        resp = await self._client.get(img_url)
        assert resp.status == 200, f"got resp [{resp.status}]: {await resp.text()}"

        logger.info(f"downloaded image {img_url=}")
        with tempfile.TemporaryFile() as fp:
            fp.write(await resp.read())
            fp.seek(0, 0)

            resp = await self._client.put(
                url=f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}/{key}", data=fp.read()
            )
            assert resp.status == 200, f"got resp [{resp.status}]: {await resp.text()}"

            logger.info(f"uploaded image {img_url=}")
            img_tmp_url = f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}/{key}"

            watermark = urlsafe_b64encode(watermark.encode("utf-8")).decode("utf-8")
            resp = await self._client.get(
                f"{img_tmp_url}?watermark/3/type/3/text/{watermark}"
            )
            assert (
                resp.status == 200
            ), f"download signed image [{resp.status}]: {await resp.text()}"

            await self._client.delete(img_tmp_url)

            return await resp.read()

    async def extract_watermark(self, imgfile) -> bytearray:
        # upload image to cloud
        fname = f"{uuid1()}.{os.path.splitext(imgfile.filename)[1]}"
        upload_key = f"/verify/{fname}"
        extract_key = f"/extracted/{fname}"
        resp = await self._client.put(
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
        assert (
            resp.status == 200
        ), f"upload image to extract watermark [{resp.status}]: {await resp.text()}"

        logger.info(f"uploaded verify image {fname=}")
        watermark_url = f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}{extract_key}"

        resp = await self._client.get(watermark_url)
        assert (
            resp.status == 200
        ), f"download watermark [{resp.status}]: {await resp.text()}"

        await wait(
            [
                self._client.delete(
                    f"{TENCENT_CLOUD_WATERMARK_BUCKET_URL}{upload_key}"
                ),
                self._client.delete(f"{watermark_url}"),
            ]
        )

        return await resp.read()


tencent_cloud = TencentCloud()
