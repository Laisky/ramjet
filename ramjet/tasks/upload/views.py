import asyncio
import codecs
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import aiohttp
import aiohttp_jinja2
from ramjet.engines import thread_executor
from ramjet.utils import logger

# DEST_DIR_PATH = "/home/laisky/test/zip"
DEST_DIR_PATH = "/opt/cwpp/prototype/oogway"


class UploadFileView(aiohttp.web.View):
    @aiohttp_jinja2.template("upload/proto.html")
    async def get(self):
        return

    async def post(self):
        data = await self.request.post()
        assert data["file"], "must post file"
        await asyncio.get_event_loop().run_in_executor(
            thread_executor, self.parse_and_update_proto, data
        )
        return aiohttp.web.HTTPFound("http://10.217.57.164:8888/云甲/")

    def parse_and_update_proto(self, post_data):
        logger.info("updating uploaded proto file")
        zip_fname = "uploaded.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_fpath = os.path.join(tmpdir, zip_fname)
            with open(zip_fpath, "wb") as fp:
                fp.write(post_data["file"].file.read())

            extract_dir = os.path.join(tmpdir, "extracted")
            with zipfile.ZipFile(zip_fpath) as fp:
                for fname in fp.namelist():
                    extract_fpath = Path(fp.extract(fname, extract_dir))

                    # zipfile 会用 cp437 对待 non-ascii 文件名，需要手动转码
                    try:
                        new_fpath = os.path.join(
                            extract_dir, fname.encode("cp437").decode("utf8")
                        )
                    except:
                        new_fpath = os.path.join(
                            extract_dir, fname.encode("cp437").decode("gbk")
                        )

                    new_dir = os.path.dirname(new_fpath)
                    if not os.path.exists(new_dir):
                        os.mkdir(new_dir)

                    extract_fpath.rename(new_fpath)

            logger.info(f"remove dir {DEST_DIR_PATH}")
            if os.path.isdir(DEST_DIR_PATH):
                shutil.rmtree(DEST_DIR_PATH)

            logger.info(f"update dir {DEST_DIR_PATH}")
            shutil.move(
                extract_dir,
                DEST_DIR_PATH,
            )
