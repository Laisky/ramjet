import asyncio
import os
import shutil
import tempfile

import aiohttp
import aiohttp_jinja2
from ramjet.archive import bounded_extract
from ramjet.engines import thread_executor
from ramjet.utils import logger

# DEST_DIR_PATH = "/home/laisky/test/zip"
DEST_DIR_PATH = "/opt/cwpp/prototype/oogway"


# One admitted upload retains its slot until background extraction finishes.
UPLOAD_SLOT = asyncio.BoundedSemaphore(1)
ARCHIVE_LIMITS = dict(
    max_compressed=100 * 1024**2,
    max_expanded=500 * 1024**2,
    max_entries=10000,
    max_ratio=1000,
    timeout=30,
)


class UploadFileView(aiohttp.web.View):
    @aiohttp_jinja2.template("upload/proto.html")
    async def get(self):
        return

    async def post(self):
        if UPLOAD_SLOT.locked():
            raise aiohttp.web.HTTPTooManyRequests(text="another upload is in progress")
        await UPLOAD_SLOT.acquire()
        submitted = False
        try:
            data = await self.request.post()
            if not isinstance(data.get("file"), aiohttp.web.FileField):
                raise aiohttp.web.HTTPBadRequest(text="must post a ZIP file")
            future = asyncio.get_running_loop().run_in_executor(
                thread_executor, self.parse_and_update_proto, data
            )

            def finished(done):
                """Release admission only after the worker completes, including cancellation."""
                UPLOAD_SLOT.release()
                if not done.cancelled():
                    done.exception()

            future.add_done_callback(finished)
            submitted = True
            try:
                await asyncio.shield(future)
            except ValueError:
                raise aiohttp.web.HTTPBadRequest(
                    text="invalid or oversized ZIP archive"
                ) from None
        finally:
            if not submitted:
                UPLOAD_SLOT.release()
        return aiohttp.web.HTTPFound("http://10.217.57.164:8888/云甲/")

    def parse_and_update_proto(self, post_data):
        logger.info("updating uploaded proto file")
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = os.path.join(tmpdir, "extracted")
            bounded_extract(post_data["file"].file, extract_dir, **ARCHIVE_LIMITS)

            logger.info(f"remove dir {DEST_DIR_PATH}")
            if os.path.isdir(DEST_DIR_PATH):
                shutil.rmtree(DEST_DIR_PATH)

            logger.info(f"update dir {DEST_DIR_PATH}")
            shutil.move(
                extract_dir,
                DEST_DIR_PATH,
            )
