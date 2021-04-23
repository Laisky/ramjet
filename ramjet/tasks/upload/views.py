import os
import shutil
import tempfile
import zipfile

import aiohttp
import aiohttp_jinja2
from ramjet.utils import logger

# DEST_DIR_PATH = "/home/laisky/test/zip"
DEST_DIR_PATH = "/opt/cwpp/prototype/oogway"


class UploadFileView(aiohttp.web.View):
    @aiohttp_jinja2.template("upload/proto.html")
    async def get(self):
        return

    async def post(self):
        zip_fname = "uploaded.zip"
        data = await self.request.post()
        assert data["file"], "must post file"
        logger.info("got uploaded proto file")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_fpath = os.path.join(tmpdir, zip_fname)
            with open(zip_fpath, "wb") as fp:
                fp.write(data["file"].file.read())

            extract_path = os.path.join(tmpdir, "extracted")
            extract_dir_name = ""
            with zipfile.ZipFile(zip_fpath) as fp:
                # 解压的文件夹名是乱码，暂时不清楚怎么处理，直接保存了拿来用
                extract_dir_name = fp.namelist()[0]
                fp.extractall(extract_path)

            logger.info(f"remove dir {DEST_DIR_PATH}")
            if os.path.isdir(DEST_DIR_PATH):
                shutil.rmtree(DEST_DIR_PATH)

            # os.mkdir(DEST_DIR_PATH)
            logger.info(f"update dir {DEST_DIR_PATH}")
            shutil.move(
                os.path.join(extract_path, extract_dir_name),
                # os.path.join(extract_path).encode("gbk"),
                DEST_DIR_PATH,
            )

        return aiohttp.web.HTTPFound("http://10.217.57.164:8888/")
