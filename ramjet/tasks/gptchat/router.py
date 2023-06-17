import asyncio
import pickle
import os
import urllib.parse
from urllib.parse import quote
import tempfile
from typing import List, Dict, Tuple
from collections import namedtuple

from aiohttp.web_request import FileField
import aiohttp.web
from minio import Minio
from minio.error import S3Error
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from ramjet.settings import prd
from ramjet.engines import thread_executor
from .auth import authenticate_by_appkey as authenticate
from .base import logger
from .embedding.query import setup, query
from .embedding.embeddings import (
    embedding_pdf,
    save_encrypt_store,
    new_store,
    load_encrypt_store,
    Index,
)
from .embedding.query import build_chain

s3cli: Minio = Minio(
    endpoint=prd.S3_MINIO_ADDR,
    access_key=prd.S3_KEY,
    secret_key=prd.S3_SECRET,
    secure=False,
)

def bind_handle(add_route):
    logger.info("bind gpt web handlers")
    setup()

    add_route("", LandingPage)
    add_route("query", Query)
    add_route("files", PDFFiles)
    add_route("ctx", EmbeddingContext)


class LandingPage(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")


class Query(aiohttp.web.View):
    """query by pre-embedded pdf files"""

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
    """build private dataset by embedding pdf files"""

    @authenticate
    async def get(self, uid):
        """list s3 files"""
        objs = []
        for obj in s3cli.list_objects(
            bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=uid,
            recursive=True,
        ):
            if obj.object_name.endswith(".store"):
                objs.append(os.path.basename(obj.object_name.removesuffix(".store")))

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

        try:
            objs = await ioloop.run_in_executor(
                thread_executor, self.process_file, uid, data
            )
        except Exception as e:
            logger.exception(f"failed to process file {data.get('file', '')}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response(
            {
                "file": objs,
            }
        )

    def process_file(self, uid, data) -> List[str]:
        file = data.get("file", "")
        assert type(file) == FileField, f"file must be FileField, got {type(file)}"

        dataset_name = data.get("file_key", "")
        assert type(dataset_name) == str, "file_key must be string"
        assert dataset_name, "file_key is required"

        data_key = data.get("data_key", "")
        assert type(data_key) == str, "data_key must be string"
        assert data_key, "data_key is required"

        logger.info(f"process file {file.filename} for {uid}")
        # write file to tmp file and delete after used
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(file.file.read())
            tmp.flush()

            index = embedding_pdf(tmp.name, dataset_name)

        # save index to temp dir
        objs = []
        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_encrypt_store(
                index,
                dirpath=tmpdir,
                name=dataset_name,
                password=data_key,
            )

            # upload index to s3
            for fpath in files:
                objkey = quote(uid + fpath.removeprefix(tmpdir))
                objs.append(objkey)
                with open(fpath, "rb") as f:
                    s3cli.fput_object(
                        bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=objkey,
                        file_path=fpath,
                    )

        return objs


UserChain = namedtuple("UserChain", ["chain", "index"])
user_embeddings_chain: Dict[str, UserChain] = {}  # uid -> UserChain

class EmbeddingContext(aiohttp.web.View):
    """build private knowledge base by consisit of embedding indices"""

    @authenticate
    async def get(self, uid):
        """talk with user's private knowledge base"""
        try:
            assert user_embeddings_chain.get(uid), "there is no context for this user"

            query = self.request.query.get("q", "").strip()
            assert query, "q is required"
        except Exception as e:
            logger.exception(f"failed to get context for {uid}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        ioloop = asyncio.get_event_loop()
        resp, refs = await ioloop.run_in_executor(
            thread_executor, self.query, uid, query
        )

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    def query(self, uid: str, query: str) -> Tuple[str, List[str]]:
        resp, refs = user_embeddings_chain[uid].chain({"question": query})
        return resp, list(set(refs))

    @authenticate
    async def post(self, uid):
        """build context by selected datasets"""
        try:
            data = await self.request.json()
            datasets = data.get("datasets", [])
            assert type(datasets) == list, "datasets must be list"
            assert datasets, "datasets is required"

            data_key = data.get("data_key", "")
            assert type(data_key) == str, "data_key must be string"
            assert data_key, "data_key is required"

            ioloop = asyncio.get_event_loop()
            index = await ioloop.run_in_executor(
                thread_executor, self.load_datasets, uid, datasets, data_key
            )
            await ioloop.run_in_executor(
                thread_executor, self.build_user_chain, uid, index
            )
        except Exception as e:
            logger.exception(f"failed to parse request body")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response(
            {"msg": "ok"},
        )

    def build_user_chain(self, uid: str, index: Index):
        if os.environ.get("OPENAI_API_TYPE", "") == "azure":
            llm = AzureChatOpenAI(
                client=None,
                deployment_name="gpt35",
                # model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )
        else:
            llm = ChatOpenAI(
                client=None,
                # model_name="gpt-3.5-turbo",
                temperature=0,
                max_tokens=1000,
                streaming=False,
            )

        user_embeddings_chain[uid] = UserChain(
            chain=build_chain(llm, index.store),
            index=index,
        )

    def load_datasets(self, uid: str, datasets: List[str], password: str) -> Index:
        """load datasets from s3"""
        store = new_store()
        for dataset in datasets:
            idx_key = f"{uid}/{dataset}.index"
            store_key = f"{uid}/{dataset}.store"

            with tempfile.TemporaryDirectory() as tmpdir:
                idx_path = os.path.join(tmpdir, idx_key)
                store_path = os.path.join(tmpdir, store_key)

                logger.debug(f"load dataset {dataset} from {tmpdir}/{uid}")
                s3cli.fget_object(
                    bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=idx_key,
                    file_path=idx_path,
                )
                s3cli.fget_object(
                    bucket_name=prd.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=store_key,
                    file_path=store_path,
                )

                store_part = load_encrypt_store(
                    dirpath=os.path.join(tmpdir, uid),
                    name=dataset,
                    password=password,
                )

            store.store.merge_from(store_part.store)

        logger.info("build private knowledge base for {uid}")
        return store
