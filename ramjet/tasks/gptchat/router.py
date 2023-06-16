import asyncio
import os
import urllib.parse
from urllib.parse import quote
import tempfile
from typing import List

from aiohttp.web_request import FileField
import aiohttp.web
from minio import Minio
from minio.error import S3Error

from ramjet.settings import prd
from ramjet.engines import thread_executor
from .auth import authenticate_by_appkey as authenticate
from .base import logger
from .embedding.query import setup, query
from .embedding.embeddings import embedding_pdf, save_encrypt_store, new_store


def bind_handle(add_route):
    logger.info("bind gpt web handlers")
    setup()

    add_route("", Index)
    add_route("query", Query)
    add_route("files", PDFFiles)


class Index(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")


class Query(aiohttp.web.View):
    async def get(self):
        project = self.request.query.get("p")
        question = urllib.parse.unquote(self.request.query.get("q", ""))

        if not project:
            return aiohttp.web.Response(text="p is required", status=400)
        if not question:
            return aiohttp.web.Response(text="q is required", status=400)

        resp = await query(project, question)
        return aiohttp.web.json_response(resp._asdict())


class PDFFiles(aiohttp.web.View):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.s3: Minio = Minio(
            endpoint="100.97.108.34:19000",
            access_key=prd.S3_KEY,
            secret_key=prd.S3_SECRET,
            secure=False,
        )

    @authenticate
    async def get(self, uid):
        """list s3 files"""
        objs = []
        for obj in self.s3.list_objects(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=uid,
            recursive=False,
        ):
            if obj.object_name.endswith(".store"):
                objs.append(obj.object_name)

        return aiohttp.web.json_response(
            {
                "files": objs,
            }
        )

    @authenticate
    async def post(self, uid):
        """Upload pdf file by form"""
        data = await self.request.post()

        ioloop = asyncio.get_event_loop()
        objs = await ioloop.run_in_executor(
            thread_executor, uid, self.process_file, data
        )

        return aiohttp.web.json_response(
            {
                "file": objs,
            }
        )

    def process_file(self, uid, data) -> List[str]:
        file = data.get("file", "")
        assert type(file) == FileField, "file must be FileField"

        dataset_name = data.get("file_key", "")
        assert type(dataset_name) == str, "file_key must be string"
        assert dataset_name, "file_key is required"

        data_key = data.get("data_key", "")
        assert type(data_key) == str, "data_key must be string"
        assert data_key, "data_key is required"

        # write file to tmp file and delete after used
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(file.file.read())
            tmp.flush()

            index = embedding_pdf(tmp.name, dataset_name)

        # save index to temp dir
        objs = []
        with tempfile.NamedTemporaryFile() as tmp:
            files = save_encrypt_store(
                index,
                dirpath=tmp.name,
                name=dataset_name,
                password=data_key,
            )

            # upload index to s3
            for fpath in files:
                objkey = quote(uid + fpath.removeprefix(tmp.name))
                objs.append(objkey)
                with open(fpath, "rb") as f:
                    self.s3.fput_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=objkey,
                        file_path=fpath,
                    )

        return objs
