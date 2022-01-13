import os
import time
from pathlib import Path
from shutil import copy2
from typing import Generator, Iterator

from kipp.options import opt
from kipp.utils import setup_logger
from ramjet.tasks.twitter.base import get_md5_hierachy_dir

logger = setup_logger("move_files", debug=True)


def main():
    start_at = time.time()
    opt.add_argument("--path", type=str, required=True)
    opt.parse_args()

    for f in gen_all_files(opt.path):
        p = Path(f)
        p = Path(p.parent, get_md5_hierachy_dir(p.name))
        p.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(p):
            logger.debug(f"file exists {p}")
            continue

        copy2(f, p)
        logger.info(f"copy {f} -> {p}")

    logger.info(f"all done, cost {time.time()-start_at:.2f}s")


def gen_all_files(dirpath: str) -> Generator[str, None, None]:
    logger.info(f"gen_all_files for {dirpath=}")
    for f in os.listdir(dirpath):
        p = os.path.join(dirpath, f)
        if os.path.isdir(p):
            logger.debug(f"skip dir {p}")
            continue

        yield p


# python -m ramjet.tasks.twitter.move_files --path=/var/www/uploads/twitter
if __name__ == "__main__":
    main()
