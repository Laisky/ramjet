"""Local regressions for GPT upload admission and nested pool starvation."""

import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import logging
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_upload(executor):
    """Load actual limiter/post methods with inert scheduling and local forms."""

    from aiohttp import web

    TooMany = web.HTTPTooManyRequests

    class Loop:
        """Model executor submission while retaining unresolved jobs."""

        def run_in_executor(self, pool, fn):
            """Submit the inert job using the existing handler's scheduler shape."""
            return executor.submit(fn)

    from aiohttp.web_request import FileField
    from aiohttp import web

    env = dict(
        FileField=FileField,
        os=__import__("os"),
        threading=threading,
        logger=logging.getLogger("local-upload"),
        user_prcess_file_sema_lock=threading.RLock(),
        user_prcess_file_sema={},
        uploaded_jobs_lock=threading.RLock(),
        uploaded_jobs=set(),
        aiohttp=NS(
            web=NS(
                HTTPTooManyRequests=TooMany,
                HTTPException=web.HTTPException,
                HTTPBadRequest=web.HTTPBadRequest,
                json_response=lambda value, **kw: value,
            )
        ),
        asyncio=NS(get_event_loop=lambda: Loop()),
        thread_executor=executor,
        partial=partial,
        functools=__import__("functools"),
    )
    tree = ast.parse((ROOT / "ramjet/tasks/gptchat/router.py").read_text())
    declarations = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "uploaded_job_slots"
            for target in node.targets
        )
    ]
    for declaration in declarations:
        exec(
            compile(ast.unparse(declaration), "<actual process capacity>", "exec"), env
        )
    limiter = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "uid_ratelimiter"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "UploadedFiles"
    )
    post = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "post"
    )
    recover_tree = ast.parse((ROOT / "ramjet/tasks/gptchat/utils.py").read_text())
    recover_fn = next(
        node
        for node in recover_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "recover"
    )
    exec(compile(ast.unparse(recover_fn), "<actual HTTP recovery>", "exec"), env)
    post.decorator_list = [ast.Name(id="recover", ctx=ast.Load())]
    source = (
        "from __future__ import annotations\n"
        + ast.unparse(limiter)
        + "\n"
        + ast.unparse(post)
    )
    exec(compile(source, "<actual upload admission>", "exec"), env)
    return env, TooMany


class Scheduler:
    """Keep fake worker Futures unresolved until a test finishes them."""

    def __init__(self):
        """Create a bounded, inert local job list."""
        self.jobs = []
        self.calls = []
        self.fail = False

    def submit(self, fn):
        """Retain a job without calling file, provider or storage code."""
        if self.fail:
            raise RuntimeError("local scheduler failure")
        future = Future()
        self.calls.append(fn)
        self.jobs.append(future)
        return future


class UploadJobTests(unittest.IsolatedAsyncioTestCase):
    """Observe actual admission lifetime without network or processing."""

    def setUp(self):
        """Build inert form and scheduler fixtures."""
        self.scheduler = Scheduler()
        self.env, self.TooMany = load_upload(self.scheduler)
        self.forms = 0

        async def form():
            """Record when a local form would retain uploaded data."""
            self.forms += 1
            return {"file_key": "fixture"}

        self.view = NS(request=NS(post=form), process_file=lambda **kw: None)
        self.user = NS(uid="fixture", is_paid=True, n_concurrent=1)

    def tearDown(self):
        """Terminate all fake jobs for deterministic local cleanup."""
        for future in self.scheduler.jobs:
            if not future.done():
                future.cancel()

    def assert_process_capacity_available(self):
        """assert_process_capacity_available verifies all eight permits are reusable."""
        slots = self.env["uploaded_job_slots"]
        acquired = []
        try:
            for _ in range(8):
                self.assertTrue(slots.acquire(blocking=False))
                acquired.append(True)
            self.assertFalse(slots.acquire(blocking=False))
        finally:
            for _ in acquired:
                slots.release()

    async def post(self, user=None):
        """Call the actual handler with an inert identity fixture."""
        return await self.env["post"](self.view, user or self.user)

    async def test_process_capacity_rejects_rotating_users_before_form(self):
        """test_process_capacity rejects the ninth distinct user before form retention."""
        for index in range(8):
            user = NS(uid=f"local-{index}", is_paid=True, n_concurrent=100)
            self.assertEqual(await self.post(user), {"status": "ok"})
        with self.assertRaises(self.TooMany) as rejected:
            await self.post(NS(uid="local-overflow", is_paid=True, n_concurrent=100))
        self.assertEqual(rejected.exception.status, 429)
        self.assertEqual(self.forms, 8)
        self.assertEqual(len(self.scheduler.jobs), 8)
        self.scheduler.jobs[0].set_result([])
        self.assertEqual(
            await self.post(NS(uid="local-reused", is_paid=True, n_concurrent=100)),
            {"status": "ok"},
        )

    async def test_user_rejection_preserves_process_capacity(self):
        """test_user_rejection returns the global slot when the user limiter rejects."""
        await self.post()
        for _ in range(10):
            with self.assertRaises(self.TooMany):
                await self.post()
        for index in range(7):
            await self.post(NS(uid=f"other-{index}", is_paid=True, n_concurrent=100))
        self.assertEqual(self.forms, 8)

    async def test_permit_covers_unresolved_worker(self):
        """Reject a second job until the first has actually completed."""
        self.assertEqual(await self.post(), {"status": "ok"})
        with self.assertRaises(self.TooMany):
            await self.post()
        self.assertEqual(self.forms, 1)
        self.scheduler.jobs[0].set_result([])
        self.assertEqual(await self.post(), {"status": "ok"})

    async def test_completion_failure_and_cancel_release_once(self):
        """Every real Future terminal state makes capacity reusable exactly once."""
        for terminal in ("success", "failure", "cancel"):
            await self.post()
            job = self.scheduler.jobs[-1]
            if terminal == "success":
                job.set_result([])
            elif terminal == "failure":
                job.set_exception(RuntimeError("local worker failure"))
            else:
                job.cancel()
            sema = self.env["user_prcess_file_sema"][self.user.uid]
            self.assertTrue(sema.acquire(blocking=False))
            sema.release()
            self.assert_process_capacity_available()
        self.assertEqual(len(self.env["uploaded_jobs"]), 0)

    async def test_submission_and_form_failures_release_capacity(self):
        """Admission is rolled back when no background worker was submitted."""
        self.scheduler.fail = True
        self.assertIn("error", await self.post())
        self.assert_process_capacity_available()
        self.scheduler.fail = False

        async def bad_form():
            """Fail before retaining any local form."""
            raise RuntimeError("local form failure")

        good_form = self.view.request.post
        self.view.request.post = bad_form
        self.assertIn("error", await self.post())
        self.assert_process_capacity_available()
        self.view.request.post = good_form
        self.assertEqual(await self.post(), {"status": "ok"})

    async def test_body_limit_status_is_preserved(self):
        """Keep aiohttp's existing HTTP body-limit response and release admission."""
        from aiohttp import web

        async def too_large():
            """Raise the framework's real synthetic body-limit exception."""
            raise web.HTTPRequestEntityTooLarge(max_size=10, actual_size=11)

        self.view.request.post = too_large
        with self.assertRaises(web.HTTPRequestEntityTooLarge):
            await self.post()
        sema = self.env["user_prcess_file_sema"][self.user.uid]
        self.assertTrue(sema.acquire(blocking=False))
        sema.release()
        self.assertEqual(self.scheduler.jobs, [])
        self.assert_process_capacity_available()

    async def test_request_file_cleanup_does_not_close_worker_file(self):
        """Give the background worker a descriptor independent of request cleanup."""
        import tempfile
        from aiohttp.web_request import FileField

        source = tempfile.TemporaryFile()
        self.addCleanup(source.close)
        source.write(b"local uploaded bytes")
        source.seek(0)

        async def form():
            """Supply a real disposable uploaded-file descriptor."""
            return {"file": FileField("file", "fixture.txt", source, "text/plain", {})}

        self.view.request.post = form
        self.view.process_file = lambda user, data: data["file"].file.read()
        await self.post()
        source.close()
        job = self.scheduler.jobs[-1]
        submitted = self.scheduler.calls[-1]
        worker_file = submitted.keywords["data"]["file"].file
        self.assertEqual(submitted(), b"local uploaded bytes")
        job.set_result([])
        self.assertTrue(worker_file.closed)

    async def test_cancelled_form_releases_existing_permit(self):
        """Cancellation before submission releases capacity without creating a job."""
        entered = asyncio.Event()
        pending = asyncio.get_running_loop().create_future()

        async def form():
            """Pause local multipart parsing until the HTTP task is cancelled."""
            entered.set()
            return await pending

        self.view.request.post = form
        task = asyncio.create_task(self.post())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        sema = self.env["user_prcess_file_sema"][self.user.uid]
        self.assertTrue(sema.acquire(blocking=False))
        sema.release()
        self.assertEqual(self.scheduler.jobs, [])
        self.assert_process_capacity_available()


class UploadedFileProcessingTests(unittest.TestCase):
    """Run actual file orchestration with local files and inert provider/storage."""

    def test_embedding_runs_directly_with_configured_provider(self):
        """The outer worker must preserve provider settings without nested submission."""
        import tempfile
        import os
        from aiohttp.web_request import FileField
        from Crypto.Cipher import AES

        calls = []
        index = object()

        def embedding_file(**kwargs):
            """Read only the local copied fixture, retaining the actual call arguments."""
            self.assertEqual(Path(kwargs["fpath"]).read_bytes(), b"fixture bytes")
            calls.append(kwargs)
            return index

        class ForbiddenExecutor:
            """Reject any inner job submission from the occupied outer worker."""

            def submit(self, *args, **kwargs):
                """Make nested scheduling an immediate bounded test failure."""
                raise AssertionError("nested worker submission")

        env = dict(
            FileField=FileField,
            os=os,
            tempfile=tempfile,
            logger=logging.getLogger("local-processing"),
            settings=NS(
                OPENAI_S3_EMBEDDINGS_PREFIX="local",
                OPENAI_EMBEDDING_FILE_SIZE_LIMIT=1024,
                OPENAI_EMBEDDING_REF_URL_PREFIX="https://fixture.invalid/",
                OPENAI_S3_CHUNK_CACHE_BUCKET="fixture",
            ),
            DEFAULT_MAX_CHUNKS_FOR_FREE=10,
            DEFAULT_MAX_CHUNKS_FOR_PAID=20,
            embedding_file=embedding_file,
            thread_executor=ForbiddenExecutor(),
            partial=partial,
            derive_key=lambda password: b"0" * 16,
            AES=AES,
            s3cli=NS(fput_object=lambda **kwargs: None),
            save_encrypt_store=lambda **kwargs: self.assertIs(kwargs["index"], index),
        )
        tree = ast.parse((ROOT / "ramjet/tasks/gptchat/router.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UploadedFiles"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_process_file"
        )
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(method),
                "<actual file orchestration>",
                "exec",
            ),
            env,
        )
        with tempfile.TemporaryFile() as source:
            source.write(b"fixture bytes")
            source.seek(0)
            data = dict(
                file=FileField("file", "fixture.txt", source, "text/plain", {}),
                file_key="fixture",
                data_key="local-password",
            )
            user = NS(
                uid="fixture",
                is_paid=True,
                apikey="local-inert-key",
                api_base="https://configured-provider.invalid/v1",
            )
            self.assertEqual(
                env["_process_file"](NS(), user, data), ["local/fixture/fixture.txt"]
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["apikey"], user.apikey)
        self.assertEqual(calls[0]["api_base"], user.api_base)
        self.assertEqual(calls[0]["max_chunks"], 20)


class EmbeddingPoolTests(unittest.TestCase):
    """Use two actual outer workers and guarantee cleanup if old code deadlocks."""

    def test_full_outer_pool_can_finish_embedding_batches(self):
        """Embedding must not synchronously wait on jobs submitted to its own pool."""
        jobs = []
        lock = threading.Lock()

        class TrackedPool(ThreadPoolExecutor):
            """Track all outer and nested work so cleanup can cancel queued inner jobs."""

            def submit(self, fn, *args, **kwargs):
                """Retain local Futures for bounded cleanup."""
                future = super().submit(fn, *args, **kwargs)
                with lock:
                    jobs.append(future)
                return future

        class Store:
            """Collect inert embeddings in stable batch order."""

            def __init__(self, values=None):
                """Create an empty local result store."""
                self.values = values or []

            def merge_from(self, other):
                """Merge a completed local batch without provider calls."""
                self.values.extend(other.values)

        pool = TrackedPool(max_workers=2)
        barrier = threading.Barrier(2)
        chunks = [NS(text=str(n), metadata={}) for n in range(12)]
        env = dict(
            logger=logging.getLogger("local-embedding"),
            thread_executor=pool,
            new_store=lambda **kw: NS(store=Store()),
            _embeddings_worker=lambda texts, **kw: Store(texts),
        )
        tree = ast.parse((ROOT / "ramjet/tasks/gptchat/llm/embeddings.py").read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "embed_chunks"
        )
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(fn),
                "<actual batch embedding>",
                "exec",
            ),
            env,
        )

        def outer():
            """Occupy both workers before executing the actual embedding helper."""
            barrier.wait(timeout=2)
            return env["embed_chunks"](chunks, "local-inert-key")

        try:
            first, second = pool.submit(outer), pool.submit(outer)
            for job in (first, second):
                self.assertEqual(
                    job.result(timeout=1).store.values, [str(n) for n in range(12)]
                )
            self.assertEqual(len(jobs), 2, "embedding unexpectedly queued nested work")
        finally:
            with lock:
                pending = list(jobs)
            for job in pending:
                job.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
