import asyncio
import base64
import hashlib
import io
import os
import re
import tempfile
import threading
import time
import urllib.parse
from functools import partial
from typing import Dict, List, Set, Tuple, Callable
from uuid import uuid1

import aiohttp.web
from aiohttp.web_request import FileField
from Crypto.Cipher import AES
from kipp.decorator import timer
from minio import Minio

from ramjet.engines import thread_executor
from ramjet import settings
from ramjet.utils import Cache

from .base import logger
from .llm.embeddings import (
    DEFAULT_MAX_CHUNKS,
    Index,
    build_user_chain,
    derive_key,
    download_chatbot_index,
    embedding_file,
    load_encrypt_store,
    new_store,
    restore_user_chain,
    save_encrypt_store,
    save_plaintext_store,
    user_embeddings_chain,
    user_embeddings_chain_mu,
    user_shared_chain,
    user_shared_chain_mu,
)
from .llm.image import (
    draw_image_by_dalle,
    upload_image_to_s3,
    image_objkey,
    draw_image_by_dalle_azure,
)
from .llm.query import (
    build_llm_for_user,
    classificate_query_type,
    query_for_prebuild_qa,
    search_for_prebuild_qa,
    setup,
)
from .llm.scan import summary_content
from .utils import authenticate_by_appkey as authenticate
from .utils import authenticate_by_appkey_sync as authenticate_sync
from .utils import recover

# limit concurrent process files by uid
user_prcess_file_sema_lock = threading.RLock()
user_prcess_file_sema: Dict[str, threading.Semaphore] = {}

# track processing files by uid
# {uid: [filekeys]}
user_processing_files_lock = threading.RLock()
user_processing_files: Dict[str, Set[str]] = {}

# valid dataset name
dataset_name_regex = re.compile(r"^[a-zA-Z0-9_-]+$")

s3cli: Minio = Minio(
    endpoint=settings.S3_MINIO_ADDR,
    access_key=settings.S3_KEY,
    secret_key=settings.S3_SECRET,
    secure=True,
)


def bind_handle(add_route):
    logger.info("bind gpt web handlers")

    # thread_executor.submit(setup)
    setup()

    add_route("", LandingPage)
    add_route("query{op:(/.*)?}", Query)
    add_route("files", UploadedFiles)
    add_route("ctx{op:(/.*)?}", EmbeddingContext)
    add_route("image{op:(/.*)?}", Image)
    add_route("encrypted-files/{filekey:.*}", EncryptedFiles)


def uid_ratelimiter(user: settings.UserPermission, concurrent=3) -> threading.Semaphore:
    logger.debug(f"uid_ratelimiter: {user.uid=}")
    uid = user.uid
    limit = user.n_concurrent or concurrent

    if user.is_paid:
        sema = user_prcess_file_sema.get(uid)
    else:
        # all free users share the same semaphore
        sema = user_prcess_file_sema.get("public")

    if not sema:
        with user_prcess_file_sema_lock:
            sema = user_prcess_file_sema.get(uid)
            if not sema:
                sema = threading.Semaphore(limit)
                user_prcess_file_sema[uid] = sema

    if not sema.acquire(blocking=False):
        raise aiohttp.web.HTTPTooManyRequests(
            reason=f"current user {uid} can only process {limit} files concurrently"
        )

    return sema


def uid_method_ratelimiter(concurrent=3):
    """rate limit by uid

    the first argument of the decorated function must be uid
    """

    def decorator(func):
        def wrapper(self, user: settings.UserPermission, *args, **kwargs):
            sema = uid_ratelimiter(user, concurrent)

            try:
                return func(self, user, *args, **kwargs)
            finally:
                sema.release()

        return wrapper

    return decorator


class LandingPage(aiohttp.web.View):
    async def get(self):
        return aiohttp.web.Response(text="welcome to gptchat")


class Image(aiohttp.web.View):
    @recover
    @authenticate
    async def post(self, user: settings.UserPermission):
        op = self.request.match_info["op"]
        data = dict(await self.request.json())
        logger.debug(f"post image api, {op=}")

        if op == "/dalle":
            task_id = str(uuid1())
            thread_executor.submit(
                self.catch_and_upload_err,
                task_id=task_id,
                func=partial(
                    self._draw_by_dalle, data=data, user=user, task_id=task_id
                ),
            )
            resp = aiohttp.web.json_response(
                {
                    "task_id": task_id,
                    "image_url": (
                        f"{settings.S3_SERVER}/"
                        f"{settings.OPENAI_S3_CHUNK_CACHE_BUCKET}/{image_objkey(task_id=task_id)}",
                    ),
                }
            )
        else:
            resp = aiohttp.web.Response(text=f"unknown op, {op=}", status=400)

        return resp

    @timer
    def catch_and_upload_err(self, task_id: str, func: Callable[[], None]):
        try:
            return func()
        except Exception as err:
            objkey = f"{os.path.splitext(image_objkey(task_id=task_id))[0]}.err.txt"
            logger.exception(f"catch and upload image drawing error, {objkey=}")
            errmsg = str(err).encode("utf-8")
            s3cli.put_object(
                bucket_name=settings.OPENAI_S3_CHUNK_CACHE_BUCKET,
                object_name=objkey,
                data=io.BytesIO(errmsg),
                length=len(errmsg),
            )

    def _draw_by_dalle(
        self,
        user: settings.UserPermission,
        data: dict,
        task_id: str,
    ) -> None:
        """draw image by openai dalle-2 and upload to s3

        Args:
            user (settings.UserPermission): user info
            data (dict): request data
        """
        prompt = data.get("prompt", "")
        assert isinstance(prompt, str), "prompt must be string"
        assert prompt, "prompt is required"

        logger.debug(f"draw image by dalle, user={user.uid}, {task_id=}")

        model_type = self.request.headers.getone("X-Laisky-Image-Token-Type", "azure")
        if model_type == "azure":
            img_content = draw_image_by_dalle_azure(prompt=prompt, apikey=user.apikey)
        elif model_type == "openai":
            img_content = draw_image_by_dalle(prompt=prompt, apikey=user.apikey)
        else:
            raise Exception(f"unknown image model type {model_type}")

        upload_image_to_s3(
            s3cli=s3cli, img_content=img_content, task_id=task_id, prompt=prompt
        )


class Query(aiohttp.web.View):
    """query by pre-embedded pdf files"""

    @recover
    @authenticate
    async def get(self, user: settings.UserPermission):
        op = self.request.match_info.get("op", "query")

        project = self.request.query.get("p")
        question = urllib.parse.unquote(self.request.query.get("q", ""))

        if not project:
            return aiohttp.web.Response(text="p is required", status=400)
        if not question:
            return aiohttp.web.Response(text="q is required", status=400)

        logger.debug(f"query prebuild qa, {op=}, {project=}, {question=}")
        ioloop = asyncio.get_running_loop()
        if op == "/query":
            resp = await ioloop.run_in_executor(
                thread_executor,
                partial(self.query, user=user, project=project, question=question),
            )
        elif op == "/search":
            resp = await ioloop.run_in_executor(
                thread_executor,
                partial(self.search, project=project, question=question),
            )
        else:
            return aiohttp.web.Response(text=f"unknown op, {op=}", status=400)

        return aiohttp.web.json_response(resp._asdict())

    @uid_method_ratelimiter()
    def query(self, user: settings.UserPermission, project: str, question: str):
        llm = build_llm_for_user(user)
        return query_for_prebuild_qa(project, question, llm)

    def search(self, project: str, question: str):
        return search_for_prebuild_qa(project, question)

    @recover
    @authenticate
    async def post(self, user):
        op = self.request.match_info["op"]
        data = dict(await self.request.json())
        if op == "/chunks":
            return await asyncio.get_event_loop().run_in_executor(
                thread_executor,
                partial(
                    _embedding_chunk_worker, request=self.request, data=data, user=user
                ),
            )
        else:
            return aiohttp.web.Response(text=f"unknown op, {op=}", status=400)


_embedding_chunk_cache = Cache(s3cli=s3cli)


def _make_embedding_chunk(
    cache_key: str,
    b64content: str,
    ext: str,
    apikey: str,
    max_chunks: int = 0,
    api_base: str = "https://api.openai.com/v1/",
) -> Tuple[Index, bool]:
    """
    Args:
        cache_key (str): cache key
        b64content (str): base64 encoded content
        ext (str): file ext, like '.html'
        apikey(str): apikey for openai api
        max_chunks (int, optional): max chunks to embed. Defaults to 0.
        api_base (str, optional): openai api base url. Defaults to "https://api.openai.com/v1/".

    Returns:
        Tuple[Index, bool]: (index, is_cached)
    """
    logger.debug(f"make embedding chunk, {cache_key=}, {ext=}, {max_chunks=}")
    idx = Index.deserialize(
        data=_embedding_chunk_cache.get_cache(cache_key),
        api_key=apikey,
    )
    if idx:
        return idx, True

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, f"content{ext}")
        with open(fpath, "wb") as fp:
            fp.write(base64.b64decode(b64content))

        idx = embedding_file(
            fpath=fpath,
            metadata_name="query",
            apikey=apikey,
            max_chunks=max_chunks,
            api_base=api_base,
        )
        _embedding_chunk_cache.save_cache(
            cache_key, idx.serialize(), expire_at=time.time() + 3600 * 24
        )
        return idx, False


def _embedding_chunk_worker(
    request: aiohttp.web.Request, data: Dict[str, str], user: settings.UserPermission
) -> aiohttp.web.Response:
    b64content = data.get("content")
    assert b64content, "base64 encoded content is required"
    user_query = data.get("query")
    assert user_query, "query is required"
    ext = data.get("ext")
    assert ext, "ext is required, like '.html'"
    model = data.get("model") or "gpt-3.5-turbo-1106"

    cache_key = hashlib.sha1(base64.b64decode(b64content)).hexdigest() + "-v2"
    max_chunks = data.get("max_chunks", 0) or DEFAULT_MAX_CHUNKS
    assert isinstance(max_chunks, int), "max_chunks must be int"
    if user.is_paid:
        max_chunks = 5000

    logger.debug(
        f"_embedding_chunk_worker for {ext=}, {model=}, {cache_key=}, {max_chunks=}, {user.info()=}"
    )

    task_type = classificate_query_type(
        query=user_query, apikey=user.apikey, api_base=user.api_base
    )
    if task_type == "search":
        return _chunk_search(
            cache_key=cache_key,
            query=user_query,
            b64content=b64content,
            ext=ext,
            apikey=user.apikey,
            api_base=user.api_base,
            max_chunks=max_chunks,
        )
    elif task_type == "scan":
        return _query_to_summary(
            cache_key=cache_key,
            query=user_query,
            b64content=b64content,
            ext=ext,
            apikey=user.apikey,
            api_base=user.api_base,
        )
    else:
        raise Exception(f"unknown task type {task_type}")


_summary_cache = Cache(s3cli=s3cli)


def _query_to_summary(
    cache_key: str,
    query: str,
    b64content: str,
    ext: str,
    apikey: str,
    api_base: str = "https://api.openai.com/v1/",
) -> aiohttp.web.Response:
    """query to summary

    Args:
        cache_key (str): cache key
        query (str): user's query
        b64content (str): base64 encoded content
        ext (str): file ext, like '.html'
        apikey (str): apikey for openai api
        api_base (str, optional): openai api base url. Defaults to "https://api.openai.com/v1/".

    Returns:
        aiohttp.web.Response: json response
    """
    cache_key = f"summary/{cache_key[:2]}/{cache_key[2:4]}/{cache_key}"
    logger.debug(f"query to summary, {query=}, {ext=}, {cache_key=}")
    start_at = time.time()
    hit_cache = False

    summary = _summary_cache.get_cache(cache_key)
    if summary and isinstance(summary, str):
        hit_cache = True
    else:
        summary = summary_content(b64content, ext, apikey=apikey, api_base=api_base)
        _summary_cache.save_cache(cache_key, summary, expire_at=time.time() + 3600 * 24)

    logger.debug(
        f"return summary, length={len(summary)}, cost={time.time() - start_at:.2f}s, "
        f"hit_cache={hit_cache}"
    )
    return aiohttp.web.json_response(
        {
            "results": summary,
            "cache_key": cache_key,
            "cached": hit_cache,
            "operator": "scan",
        }
    )


def _chunk_search(
    cache_key: str,
    query: str,
    b64content: str,
    ext: str,
    apikey: str,
    max_chunks: int = 0,
    api_base: str = "https://api.openai.com/v1/",
) -> aiohttp.web.Response:
    """search in embedding chunk

    Args:
        cache_key (str): cache key
        query (str): user's query
        content (str): base64 encoded content
        ext (str): file ext, like '.html'
        apikey (str): apikey for openai api
        max_chunks (int, optional): max chunks to embed. Defaults to 0.
        api_base (str, optional): openai api base url. Defaults to "https://api.openai.com/v1/".

    Returns:
        aiohttp.web.Response: json response
    """
    cache_key = f"chunksearch/{cache_key[:2]}/{cache_key[2:4]}/{cache_key}"
    logger.debug(f"search embedding chunk, {query=}, {ext=}, {api_base=}, {cache_key=}")
    start_at = time.time()
    idx, cached = _make_embedding_chunk(
        cache_key=cache_key,
        b64content=b64content,
        ext=ext,
        apikey=apikey,
        api_base=api_base,
        max_chunks=max_chunks,
    )
    refs = idx.store.similarity_search(query, k=5)
    results = "\n".join([ref.page_content for ref in refs if ref.page_content])
    logger.info(
        f"return similarity search results, length={len(results)}, "
        f"cost={time.time() - start_at:.2f}s"
    )
    return aiohttp.web.json_response(
        {
            "results": results,
            "cache_key": cache_key,
            "cached": cached,
            "operator": "search",
        }
    )


class EncryptedFiles(aiohttp.web.View):
    @recover
    async def get(self):
        # https://uid:password@fikekey.pdf
        # get uid and password from request basic auth
        auth_header = self.request.headers.get("Authorization", "").strip()
        auth_header = auth_header.removeprefix("Basic").removeprefix("Bearer").strip()
        if auth_header:
            decoded_credentials = base64.b64decode(auth_header).decode("utf-8")
            _, password = decoded_credentials.split(":")
        else:
            response = aiohttp.web.Response(status=401)
            response.headers["WWW-Authenticate"] = 'Basic realm="My Realm"'
            return response

        # download pdf file to temp dir
        filekey = self.request.match_info["filekey"]
        filename = os.path.basename(filekey)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, filename)

            # fix bug for https://github.com/minio/minio-py/pull/1309
            # tmp_file = os.path.join(tmpdir, "tmp.part.minio")

            # s3cli.fget_object(
            #     bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
            #     object_name=f"{filekey}",
            #     file_path=fpath,
            #     request_headers={"Cache-Control": "no-cache"},
            #     tmp_file_path=tmp_file,
            # )

            # weird bug, my minio root account cannot fget_object, tell me 403.
            # so I use aiohttp to download file
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url=f"https://s3.laisky.com/{settings.OPENAI_S3_EMBEDDINGS_BUCKET}/{filekey}",
                ) as resp:
                    assert resp.status == 200, f"failed to download file {filekey}"
                    with open(fpath, "wb") as fp:
                        fp.write(await resp.read())

            # decrypt pdf file
            with open(fpath, "rb") as f:
                try:
                    key = derive_key(password)
                    nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
                    cipher = AES.new(key, AES.MODE_EAX, nonce)
                    data = cipher.decrypt_and_verify(ciphertext, tag)
                except Exception:
                    logger.exception(f"failed to decrypt file {filekey}, ask to retry")
                    response = aiohttp.web.Response(status=401)
                    response.headers["WWW-Authenticate"] = 'Basic realm="My Realm"'
                    return response

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == ".pdf":
            content_type = "application/pdf"
        elif file_ext == ".doc" or file_ext == ".docx":
            content_type = "application/msword"
        elif file_ext == ".xls" or file_ext == ".xlsx":
            content_type = "application/vnd.ms-excel"
        elif file_ext == ".ppt" or file_ext == ".pptx":
            content_type = "application/vnd.ms-powerpoint"
        elif file_ext == ".md":
            content_type = "text/markdown"
        else:
            content_type = "application/octet-stream"

        # Add charset to headers
        headers = {"Content-Type": content_type + "; charset=utf-8"}

        # return decrypted pdf file
        return aiohttp.web.Response(body=data, headers=headers)


class UploadedFiles(aiohttp.web.View):
    """build private dataset by embedding files

    Returns:
        aiohttp.web.Response -- json response, contains file status
        ::
            [
                {"name": "file1", "status": "done"},
                {"name": "file2", "status": "processing", "progress": 75},
            ]
    """

    @recover
    @authenticate
    async def get(self, user: settings.UserPermission):
        """list s3 files"""
        uid = user.uid
        files: List[Dict] = []
        password = self.request.headers.get("X-PDFCHAT-PASSWORD", "")

        # for test
        # files.append({"name": "test.pdf", "status": "processing", "progress": 75})

        for obj in s3cli.list_objects(
            bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/",
            recursive=False,
        ):
            if obj.object_name.endswith(".store"):
                files.append(
                    {
                        "name": os.path.basename(
                            obj.object_name.removesuffix(".store")
                        ),
                        "status": "done",
                    }
                )

        with user_processing_files_lock:
            files.extend(
                [
                    {"name": f, "status": "processing", "progress": 75}
                    for f in user_processing_files.get(uid, [])
                ]
            )

        selected = []
        try:
            with user_embeddings_chain_mu:
                need_restore = uid not in user_embeddings_chain

            if need_restore:
                restore_user_chain(s3cli, user, password)

            with user_embeddings_chain_mu:
                if uid in user_embeddings_chain:
                    selected = user_embeddings_chain[uid].datasets
        except Exception as err:
            logger.debug(f"failed to restore user chain {uid}, {err=}")

        return aiohttp.web.json_response(
            {
                "datasets": files,
                "selected": selected,
            }
        )

    @recover
    @authenticate
    async def delete(self, user: settings.UserPermission):
        """delete s3 files"""
        data = await self.request.json()
        datasets = data.get("datasets", [])
        if not datasets:
            return aiohttp.web.json_response(
                {"error": "datasets is required"}, status=400
            )

        assert isinstance(datasets, list), "datasets must be a array"

        try:
            ioloop = asyncio.get_event_loop()

            # do not wait task done
            ioloop.run_in_executor(
                thread_executor,
                partial(self.delete_files, user=user, dataset_names=datasets),
            )
        except Exception as e:
            logger.exception(f"failed to delete files {data.get('files', [])}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)

        return aiohttp.web.json_response({"status": "ok"})

    def delete_files(self, user: settings.UserPermission, dataset_names: List[str]):
        uid = user.uid
        for ext in ["store", "index", "pdf"]:
            for dataset_name in dataset_names:
                objkey = (
                    f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset_name}.{ext}"
                )
                try:
                    s3cli.remove_object(
                        bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
                        object_name=objkey,
                    )
                    logger.info(f"deleted file on s3, {objkey=}")
                except Exception:
                    logger.exception(f"failed to delete file on s3, {objkey=}")

    @recover
    @authenticate
    async def post(self, user: settings.UserPermission):
        """Upload pdf file by form"""
        data = await self.request.post()

        sema = uid_ratelimiter(user, 3)
        try:
            ioloop = asyncio.get_event_loop()

            # do not wait task done
            ioloop.run_in_executor(
                thread_executor, partial(self.process_file, user=user, data=data)
            )
        except Exception as e:
            logger.exception(f"failed to process file {data.get('file', '')}")
            return aiohttp.web.json_response({"error": str(e)}, status=400)
        finally:
            sema.release()

        return aiohttp.web.json_response({"status": "ok"})

    @timer
    def process_file(self, user: settings.UserPermission, data) -> List[str]:
        dataset_name = data.get("file_key", "")
        assert isinstance(dataset_name, str), "file_key must be string"
        assert dataset_name, "file_key is required"
        assert dataset_name_regex.match(
            dataset_name
        ), "file_key should only contain [a-zA-Z0-9_-]"

        uid = user.uid
        logger.debug(f"process file {dataset_name=} for user {uid=}")
        try:
            with user_processing_files_lock:
                fset = user_processing_files.get(uid, set())
                fset.add(dataset_name)
                user_processing_files[uid] = fset

            logger.info(f"process file {dataset_name=} for user {uid=}")
            return self._process_file(user, data)
        finally:
            with user_processing_files_lock:
                user_processing_files[uid].remove(dataset_name)

    def _process_file(self, user: settings.UserPermission, data: Dict) -> List[str]:
        """process user uploaded file by openai embeddings,
        then encrypt and upload to s3
        """
        file = data.get("file", "")
        assert isinstance(file, FileField), f"file must be FileField, got {type(file)}"

        dataset_name = data.get("file_key", "")
        password = data.get("data_key", "")
        assert isinstance(password, str), "data_key must be string"
        assert password, "data_key is required"

        max_chunks = data.get("max_chunks", DEFAULT_MAX_CHUNKS)
        assert isinstance(max_chunks, int), "max_chunks must be int"

        file_ext = os.path.splitext(file.filename)[1]

        uid = user.uid
        objs = []
        encrypted_file_key = (
            f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset_name}{file_ext}"
        )

        logger.info(f"process file {file.filename} for {uid}")
        with tempfile.TemporaryDirectory() as tmpdir:
            # copy origin uploaded file to temp dir
            source_fpath = os.path.join(tmpdir, f"_source_{dataset_name}{file_ext}")
            with open(source_fpath, "wb") as fp:
                total_size = 0
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = file.file.read(chunk_size)
                    if not chunk:
                        break

                    total_size += len(chunk)
                    assert (
                        total_size < settings.OPENAI_EMBEDDING_FILE_SIZE_LIMIT
                    ), f"file size should not exceed {settings.OPENAI_EMBEDDING_FILE_SIZE_LIMIT} bytes"

                    fp.write(chunk)

            metadata_name = (
                f"{settings.OPENAI_EMBEDDING_REF_URL_PREFIX}{encrypted_file_key}"
            )
            file_ext = os.path.splitext(fp.name)[1]

            # index = embedding_file(fp.name, metadata_name)
            index = thread_executor.submit(
                partial(
                    embedding_file,
                    fpath=fp.name,
                    metadata_name=metadata_name,
                    apikey=user.apikey,
                    max_chunks=max_chunks,
                    api_base=user.api_base,
                )
            ).result()

            # encrypt and upload origin pdf file
            encrypted_file_path = os.path.join(tmpdir, dataset_name + file_ext)
            logger.debug(f"try to upload {encrypted_file_path}")
            with open(source_fpath, "rb") as src_fp:
                with open(encrypted_file_path, "wb") as encrypted_fp:
                    key = derive_key(password)
                    cipher = AES.new(key, AES.MODE_EAX)
                    ciphertext, tag = cipher.encrypt_and_digest(src_fp.read())
                    for x in (cipher.nonce, tag, ciphertext):
                        encrypted_fp.write(x)

            objs.append(encrypted_file_key)
            s3cli.fput_object(
                bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=encrypted_file_key,
                file_path=encrypted_file_path,
            )
            logger.info(
                f"succeed upload encrypted source file {encrypted_file_key} to s3"
            )

            # encrypt and upload index
            save_encrypt_store(
                s3cli=s3cli,
                user=user,
                index=index,
                name=dataset_name,
                password=password,
            )

        return objs


class EmbeddingContext(aiohttp.web.View):
    """build private knowledge base by consisit of embedding indices"""

    @recover
    # @authenticate
    async def get(self):
        """
        dispatch request to different handlers by url
        """
        op = self.request.match_info["op"]
        logger.debug(f"embedding context, {op=}")

        ioloop = asyncio.get_event_loop()
        if op == "/chat":
            resp = await ioloop.run_in_executor(thread_executor, self.chatbot_query)
        elif op == "/search":
            resp = await ioloop.run_in_executor(thread_executor, self.chatbot_search)
        elif op == "/list":
            resp = await ioloop.run_in_executor(thread_executor, self.list_chatbots)
        elif op == "/share":
            resp = await ioloop.run_in_executor(
                thread_executor, self.share_chatbot_query
            )
        else:
            raise Exception(f"unknown op {op}")

        return resp

    @authenticate_sync
    def share_chatbot_query(
        self, user: settings.UserPermission
    ) -> aiohttp.web.Response:
        """talk with somebody shared private knowledge base"""
        # get uid and chatbot_name from query args
        chatbot_name = self.request.query.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9\-_]+$", chatbot_name), "chatbot_name is required"

        query = self.request.query.get("q", "").strip()
        assert query, "q is required"

        with user_shared_chain_mu:
            need_restore = user.uid not in user_shared_chain

        if need_restore:
            logger.debug(f"try restore user shared chain from s3 for {user.uid=}")
            restore_user_chain(s3cli=s3cli, user=user, chatbot_name=chatbot_name)

        with user_shared_chain_mu:
            chatbot = user_shared_chain[user.uid + chatbot_name]

        llm = build_llm_for_user(user)
        sema = uid_ratelimiter(user=user)
        try:
            resp, refs = chatbot.chain(llm, query)
            refs = list(set(refs))
        finally:
            sema.release()

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @authenticate_sync
    def chatbot_search(self, user: settings.UserPermission) -> aiohttp.web.Response:
        uid = user.uid
        user_query = self.request.query.get("q", "").strip()
        assert user_query, "q is required"

        password = self.request.headers.getone("X-PDFCHAT-PASSWORD")
        assert password, "X-PDFCHAT-PASSWORD is required"

        with user_embeddings_chain_mu:
            need_restore = uid not in user_embeddings_chain

        if need_restore:
            logger.debug(f"try restore user chain from s3 for {uid=}")
            restore_user_chain(s3cli, user, password)

        with user_embeddings_chain_mu:
            chatbot = user_embeddings_chain[uid]

        sema = uid_ratelimiter(user=user)
        try:
            resp, refs = chatbot.search(user_query)
            refs = list(set(refs))
        finally:
            sema.release()

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @authenticate_sync
    def chatbot_query(self, user: settings.UserPermission) -> aiohttp.web.Response:
        """talk with user's private knowledge base"""
        uid = user.uid
        user_query = self.request.query.get("q", "").strip()
        assert user_query, "q is required"

        password = self.request.headers.getone("X-PDFCHAT-PASSWORD")
        assert password, "X-PDFCHAT-PASSWORD is required"

        with user_embeddings_chain_mu:
            need_restore = uid not in user_embeddings_chain

        if need_restore:
            logger.debug(f"try restore user chain from s3 for {uid=}")
            restore_user_chain(s3cli, user, password)

        with user_embeddings_chain_mu:
            chatbot = user_embeddings_chain[uid]

        llm = build_llm_for_user(user)
        sema = uid_ratelimiter(user=user)
        try:
            resp, refs = chatbot.chain(llm, user_query)
            refs = list(set(refs))
        finally:
            sema.release()

        return aiohttp.web.json_response(
            {
                "text": resp,
                "url": refs,
            }
        )

    @authenticate_sync
    def list_chatbots(self, user: settings.UserPermission) -> aiohttp.web.Response:
        chatbot_names = []
        for obj in s3cli.list_objects(
            bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
            prefix=f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{user.uid}/chatbot/",
            recursive=False,
        ):
            if not obj.object_name.endswith(".store"):
                continue

            chatbot_names.append(
                os.path.basename(obj.object_name).removesuffix(".store")
            )

        # get current
        resp = None
        try:
            resp = s3cli.get_object(
                bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
                object_name=f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{user.uid}/chatbot/__CURRENT",
            )
            current_chatbot = resp.data.decode("utf-8")
        except Exception:
            current_chatbot = ""
        finally:
            if resp:
                resp.close()
                resp.release_conn()

        return aiohttp.web.json_response(
            {
                "current": current_chatbot,
                "chatbots": chatbot_names,
            }
        )

    @recover
    @authenticate
    async def post(self, user: settings.UserPermission):
        data = await self.request.json()

        ioloop = asyncio.get_event_loop()
        op = self.request.match_info["op"]
        logger.debug(f"post to EmbeddingContext with {op=}")
        if op == "/build":
            return await ioloop.run_in_executor(
                thread_executor,
                partial(
                    self.build_chatbot,
                    user=user,
                    data=data,
                ),
            )
        elif op == "/active":
            return await ioloop.run_in_executor(
                thread_executor,
                partial(
                    self.active_chatbot,
                    user=user,
                    data=data,
                ),
            )
        elif op == "/share":
            return await ioloop.run_in_executor(
                thread_executor,
                partial(
                    self.share_chatbot,
                    user=user,
                    data=data,
                ),
            )
        else:
            raise NotImplementedError(f"unknown op {op}")

    def share_chatbot(
        self, user: settings.UserPermission, data: Dict
    ) -> aiohttp.web.Response:
        """
        share chatbot to other users

        will decrypt user's embeddings store and save it in plain text
        """
        password = data.get("data_key", "")
        assert isinstance(password, str), "data_key must be string"
        assert password, "data_key is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        with tempfile.TemporaryDirectory() as tmpdir:
            _, index, datasets = download_chatbot_index(
                s3cli=s3cli,
                dirpath=tmpdir,
                user=user,
                chatbot_name=chatbot_name,
                password=password,
            )
            save_plaintext_store(
                s3cli=s3cli,
                user=user,
                index=index,
                datasets=datasets,
                name=chatbot_name,
            )

        logger.info(f"succeed share {user.uid}'s chatbot {chatbot_name}")
        return aiohttp.web.json_response(
            {
                "uid": user.uid,
                "chatbot_name": chatbot_name,
            }
        )

    def active_chatbot(
        self,
        user: settings.UserPermission,
        data: Dict,
    ) -> aiohttp.web.Response:
        """active existed chatbot by name"""
        password = data.get("data_key", "")
        assert isinstance(password, str), "data_key must be string"
        assert password, "data_key is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        restore_user_chain(s3cli, user, password, chatbot_name)

        return aiohttp.web.json_response(
            {"msg": "ok"},
        )

    def build_chatbot(
        self, user: settings.UserPermission, data: Dict
    ) -> aiohttp.web.Response:
        """build chatbot by selected datasets"""
        datasets = data.get("datasets", [])
        assert isinstance(datasets, list), "datasets must be list"
        assert datasets, "datasets is required"

        chatbot_name = data.get("chatbot_name", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", chatbot_name), "chatbot_name is invalid"

        password = data.get("data_key", "")
        assert isinstance(password, str), "data_key must be string"
        assert password, "data_key is required"

        self._build_user_chatbot(
            user=user,
            password=password,
            datasets=datasets,
            chatbot_name=chatbot_name,
        )
        return aiohttp.web.json_response(
            {"msg": f"{chatbot_name} build ok"},
        )

    def _build_user_chatbot(
        self,
        user: settings.UserPermission,
        password: str,
        datasets: List[str],
        chatbot_name: str = "default",
    ) -> None:
        """build user custom chatbot by selected datasets

        Args:
            user (settings.UserPermission): user
            password (str): password for decrypt
            datasets (List[str]): dataset names
            chatbot_name (str, optional): chatbot name. Defaults to "default".
        """
        uid = user.uid
        index = self.load_datasets(
            user=user,
            datasets=datasets,
            password=password,
        )
        chain = build_user_chain(
            index=index,
            datasets=datasets,
        )
        with user_embeddings_chain_mu:
            user_embeddings_chain[uid] = chain

        save_encrypt_store(
            s3cli=s3cli,
            user=user,
            index=index,
            name=chatbot_name,
            password=password,
            datasets=datasets,
        )

        # save current chatbot
        current_chatbot_objkey = (
            f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/chatbot/__CURRENT"
        )
        s3cli.put_object(
            bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
            object_name=current_chatbot_objkey,
            data=io.BytesIO(chatbot_name.encode("utf-8")),
            length=len(chatbot_name.encode("utf-8")),
        )
        logger.debug(
            f"save current chatbot {chatbot_name=}, {current_chatbot_objkey}, {uid=}"
        )

    def load_datasets(
        self,
        user: settings.UserPermission,
        datasets: List[str],
        password: str,
    ) -> Index:
        """load datasets from s3

        Args:
            user (settings.UserPermission): user
            datasets (List[str]): dataset names
            password (str): password for decrypt
        """
        uid = user.uid
        store = new_store(apikey=user.apikey, api_base=user.api_base)
        for dataset in datasets:
            idx_key = f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset}.index"
            store_key = f"{settings.OPENAI_S3_EMBEDDINGS_PREFIX}/{uid}/{dataset}.store"

            with tempfile.TemporaryDirectory() as tmpdir:
                idx_path = os.path.join(tmpdir, "dataset.index")
                store_path = os.path.join(tmpdir, "dataset.store")

                # fix bug for https://github.com/minio/minio-py/pull/1309
                tmp_file = os.path.join(tmpdir, "tmp.part.minio")

                logger.debug(f"download dataset {dataset=}, {idx_path=}")
                s3cli.fget_object(
                    bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=idx_key,
                    file_path=idx_path,
                    tmp_file_path=tmp_file,
                    # request_headers={"Cache-Control": "no-cache"},
                )
                s3cli.fget_object(
                    bucket_name=settings.OPENAI_S3_EMBEDDINGS_BUCKET,
                    object_name=store_key,
                    file_path=store_path,
                    tmp_file_path=tmp_file,
                    # request_headers={"Cache-Control": "no-cache"},
                )

                store_part = load_encrypt_store(
                    dirpath=os.path.dirname(idx_path),
                    name="dataset",
                    password=password,
                )

            store.store.merge_from(store_part.store)

        logger.info(f"build private knowledge base for {uid}")
        return store
