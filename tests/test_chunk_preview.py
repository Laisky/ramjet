import io
import os
import tempfile
import unittest

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from ramjet import settings

if not hasattr(settings, "S3_MINIO_ADDR"):
    setattr(settings, "S3_MINIO_ADDR", "localhost:9000")
if not hasattr(settings, "S3_KEY"):
    setattr(settings, "S3_KEY", "test-key")
if not hasattr(settings, "S3_SECRET"):
    setattr(settings, "S3_SECRET", "test-secret")
if not hasattr(settings, "OPENAI_S3_CHUNK_CACHE_BUCKET"):
    setattr(settings, "OPENAI_S3_CHUNK_CACHE_BUCKET", "chunks")
if not hasattr(settings, "OPENAI_S3_EMBEDDINGS_PREFIX"):
    setattr(settings, "OPENAI_S3_EMBEDDINGS_PREFIX", "embeddings")
if not hasattr(settings, "OPENAI_EMBEDDING_FILE_SIZE_LIMIT"):
    setattr(settings, "OPENAI_EMBEDDING_FILE_SIZE_LIMIT", 10 * 1024 * 1024)

from ramjet.tasks.gptchat.llm.embeddings import chunk_file, DEFAULT_MAX_CHUNKS_FOR_FREE
from ramjet.tasks.gptchat.router import ChunkPreview


def _auth_headers(**overrides):
    headers = {
        "Authorization": "Bearer test-api-key",
        "X-Laisky-User-Id": "user-1",
    }
    headers.update(overrides)
    return headers


class ChunkHelperTests(unittest.TestCase):
    def test_chunk_file_adds_source_metadata(self):
        html_body = """<html><body><p>Hello chunk</p></body></html>"""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(html_body.encode("utf-8"))
            fpath = tmp.name
        try:
            chunks = chunk_file(
                fpath=fpath,
                metadata_name="sample-doc",
                max_chunks=10,
            )
        finally:
            os.unlink(fpath)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("source", chunk.metadata)
            self.assertTrue(chunk.metadata["source"].startswith("sample-doc"))


class ChunkEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = web.Application()
        app.router.add_view("/chunks", ChunkPreview)
        self._server = TestServer(app)
        await self._server.start_server()
        self._client = TestClient(self._server)
        await self._client.start_server()

    async def asyncTearDown(self):
        await self._client.close()
        await self._server.close()

    async def test_post_plain_text(self):
        payload = {
            "text": "Hello world, this is a chunking test.",
            "metadata_name": "inline",
            "chunk_size": 1024,
            "chunk_overlap": 0,
        }
        headers = _auth_headers(**{"X-Laisky-User-Is-Free": "true"})

        resp = await self._client.post("/chunks", json=payload, headers=headers)
        self.assertEqual(resp.status, 200)
        body = await resp.json()

        self.assertEqual(body["origin"], "text")
        self.assertEqual(body["source"], "inline")
        self.assertEqual(body["chunk_size"], 1024)
        self.assertEqual(body["chunk_overlap"], 0)
        self.assertEqual(body["max_chunks"], DEFAULT_MAX_CHUNKS_FOR_FREE)
        self.assertGreaterEqual(body["total"], 1)
        texts = " ".join(chunk["text"] for chunk in body["chunks"])
        self.assertIn("chunking test", texts)

    async def test_post_file_upload(self):
        form = FormData()
        form.add_field(
            "file",
            io.BytesIO(b"Chunk api accepts files"),
            filename="sample.txt",
            content_type="text/plain",
        )
        form.add_field("metadata_name", "sample-dataset")
        form.add_field("chunk_size", "16")
        form.add_field("chunk_overlap", "0")
        headers = _auth_headers(**{"X-Laisky-User-Is-Free": "false"})

        resp = await self._client.post("/chunks", data=form, headers=headers)
        self.assertEqual(resp.status, 200)
        body = await resp.json()

        self.assertEqual(body["origin"], "file")
        self.assertEqual(body["source"], "sample-dataset")
        self.assertGreater(body["total"], 0)
        self.assertTrue(all("source" in chunk["metadata"] for chunk in body["chunks"]))
