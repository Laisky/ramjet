"""Bounded local regressions for request bodies and prototype ZIP publication."""

import ast
import asyncio
import io
import logging
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
import zipfile
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def archive(entries):
    """archive returns a small synthetic ZIP entirely in memory."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as output:
        for name, content in entries:
            output.writestr(name, content)
    stream.seek(0)
    return stream


def load_parser(destination, limits):
    """load_parser loads the actual publication method with a disposable destination."""
    env = dict(
        asyncio=asyncio,
        os=__import__("os"),
        shutil=shutil,
        tempfile=tempfile,
        zipfile=zipfile,
        Path=Path,
        logger=logging.getLogger("upload-local-tests"),
        DEST_DIR_PATH=str(destination),
        ARCHIVE_LIMITS=limits,
    )
    helper = ROOT / "ramjet/archive.py"
    if helper.exists():
        spec = importlib.util.spec_from_file_location("archive_under_test", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env["bounded_extract"] = module.bounded_extract
    tree = ast.parse((ROOT / "ramjet/tasks/upload/views.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "UploadFileView"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "parse_and_update_proto"
    )
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]), "<actual upload parser>", "exec"
        ),
        env,
    )
    return env["parse_and_update_proto"]


class UploadBoundsTests(unittest.TestCase):
    """UploadBoundsTests checks limits before replacing any published prototype."""

    def setUp(self):
        """setUp creates only disposable prototype and synthetic ZIP fixtures."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.destination = Path(self.temp.name) / "published"
        self.destination.mkdir()
        (self.destination / "existing.txt").write_text("preserve")
        self.limits = dict(
            max_compressed=4096,
            max_expanded=1024,
            max_entries=10,
            max_ratio=1000,
            timeout=10,
        )

    def parse(self, stream, **overrides):
        """parse executes the actual handler publication method against local storage."""
        parser = load_parser(self.destination, dict(self.limits, **overrides))
        parser(NS(), {"file": NS(file=stream)})

    def rejected(self, entries, **limits):
        """rejected requires a bounded failure while preserving the previous publication."""
        with self.assertRaises(ValueError):
            self.parse(archive(entries), **limits)
        self.assertEqual((self.destination / "existing.txt").read_text(), "preserve")

    def test_expansion_budget_prevents_publication(self):
        """test_expansion_budget_prevents_publication limits a tiny highly compressed archive."""
        self.rejected([("big.txt", "x" * 8192)])

    def test_entry_budget_prevents_publication(self):
        """test_entry_budget_prevents_publication bounds tiny-entry fanout."""
        self.rejected([(f"{n}.txt", "x") for n in range(3)], max_entries=2)

    def test_compressed_budget_prevents_publication(self):
        """test_compressed_budget_prevents_publication bounds copied input bytes."""
        self.rejected([("safe.txt", "safe")], max_compressed=32)

    def test_unsafe_paths_preserve_previous_publication(self):
        """test_unsafe_paths_preserve_previous_publication rejects traversal and absolute paths."""
        for name in (
            "../outside.txt",
            "/outside.txt",
            "C:/outside.txt",
            "a\\..\\outside.txt",
        ):
            with self.subTest(name=name):
                self.rejected([(name, "local")])

    def test_normal_files_directories_and_utf8_names_publish(self):
        """test_normal_files_directories_and_utf8_names_publish preserves intended ZIP contents."""
        self.parse(
            archive(
                [
                    ("assets/", ""),
                    ("assets/safe.txt", "safe"),
                    ("notes-☃.txt", "unicode"),
                ]
            )
        )
        self.assertEqual((self.destination / "assets/safe.txt").read_text(), "safe")
        self.assertEqual((self.destination / "notes-☃.txt").read_text(), "unicode")
        self.assertFalse((self.destination / "existing.txt").exists())

    def test_application_body_limit_is_100_mib(self):
        """test_application_body_limit_is_100_mib evaluates the actual constructor without allocating a body."""
        tree = ast.parse((ROOT / "ramjet/__main__.py").read_text())
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "Application"
        )
        limit = next(k.value for k in call.keywords if k.arg == "client_max_size")
        actual = eval(
            compile(ast.Expression(body=limit), "<actual request limit>", "eval")
        )
        self.assertEqual(actual, 100 * 1024**2)

    def test_ratio_timeout_and_symlinks_preserve_publication(self):
        """Reject unsafe entry types and exhausted extraction budgets."""
        self.rejected([("compressible.txt", "a" * 500)], max_ratio=2)
        self.rejected([("safe.txt", "safe")], timeout=-1)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as output:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            output.writestr(info, "outside")
        stream.seek(0)
        with self.assertRaises(ValueError):
            self.parse(stream)
        self.assertTrue((self.destination / "existing.txt").exists())

    def test_duplicates_and_invalid_zip_preserve_publication(self):
        """Reject colliding entries and malformed input before publication."""
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", UserWarning)
            self.rejected([("same.txt", "one"), ("same.txt", "two")])
        with self.assertRaises(ValueError):
            self.parse(io.BytesIO(b"not a ZIP"))
        self.assertTrue((self.destination / "existing.txt").exists())

    def test_legacy_gbk_names_publish(self):
        """Preserve the uploader's existing GBK filename compatibility."""
        raw = archive([("stub.txt", "safe")]).getvalue()
        raw = raw.replace(b"stub.txt", "中文.txt".encode("gbk"))
        self.parse(io.BytesIO(raw))
        self.assertEqual((self.destination / "中文.txt").read_text(), "safe")


class UploadAdmissionTests(unittest.IsolatedAsyncioTestCase):
    """Hold one bounded background job even after its HTTP waiter is cancelled."""

    def load_post(self, executor, slot):
        """Extract the actual post handler without importing deployment settings."""
        from aiohttp import web

        tree = ast.parse((ROOT / "ramjet/tasks/upload/views.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UploadFileView"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "post"
        )
        env = dict(
            asyncio=asyncio,
            aiohttp=NS(web=web),
            thread_executor=executor,
            UPLOAD_SLOT=slot,
        )
        exec(
            compile(
                ast.Module(body=[fn], type_ignores=[]),
                "<actual upload admission>",
                "exec",
            ),
            env,
        )
        return env["post"], web

    async def test_cancelled_waiter_keeps_worker_admission_slot(self):
        """An interrupted request cannot admit another ZIP while extraction runs."""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from aiohttp import web

        started, release = Event(), Event()
        executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=True)
        self.addCleanup(release.set)
        slot = asyncio.BoundedSemaphore(1)
        post, _ = self.load_post(executor, slot)
        stream = io.BytesIO(b"fixture")
        file = web.FileField("file", "fixture.zip", stream, "application/zip", {})

        async def body():
            """Return an inert local multipart file fixture."""
            return {"file": file}

        def worker(data):
            """Keep a single local worker alive until explicitly released."""
            started.set()
            release.wait(5)

        view = NS(request=NS(post=body), parse_and_update_proto=worker)
        task = asyncio.create_task(post(view))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            self.assertTrue(started.is_set())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(web.HTTPTooManyRequests):
                await post(view)
            self.assertTrue(slot.locked())
        finally:
            release.set()
            for _ in range(100):
                if not slot.locked():
                    break
                await asyncio.sleep(0.001)
        self.assertFalse(slot.locked())
        view.parse_and_update_proto = lambda data: None
        self.assertEqual((await post(view)).status, 302)

    async def test_invalid_form_releases_slot(self):
        """A rejected multipart form leaves the next upload admissible."""
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=True)
        slot = asyncio.BoundedSemaphore(1)
        post, web = self.load_post(executor, slot)

        async def body():
            """Return a local form without a file."""
            return {"file": "string"}

        with self.assertRaises(web.HTTPBadRequest):
            await post(NS(request=NS(post=body)))
        self.assertFalse(slot.locked())


if __name__ == "__main__":
    unittest.main()
