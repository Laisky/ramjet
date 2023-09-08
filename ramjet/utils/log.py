import logging
import multiprocessing

from kipp.options import opt
from kipp.utils import setup_logger

from ramjet.engines import thread_executor
from ramjet.settings import LOG_NAME, LOG_PATH, MAIL_FROM_ADDR, MAIL_TO_ADDRS

class MultiProcessLogHandler(logging.Handler):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        self.setup_queue()
        thread_executor.submit(self.run_dispatcher)

    def get_email_sender(self):
        return opt.email_sender

    def emit(self, record):
        try:
            ei = record.exc_info
            if ei:
                _ = self.format(
                    record
                )  # just to get traceback text into record.exc_text
                record.exc_info = None  # not needed any more
            self.queue.put_nowait(record)
        except Exception:
            self.handleError(record)

    def run_dispatcher(self):
        while 1:
            record = self.queue.get()
            if not record:
                return

            # logging.getLogger().handle(record)
            # if record.levelno > logging.WARNING:
            #     thread_executor.submit(
            #         self.get_email_sender().send_email,
            #         mail_to=MAIL_TO_ADDRS,
            #         mail_from=MAIL_FROM_ADDR,
            #         subject="Ramjet error alert",
            #         content="{}\n{}".format(record.message, record.exc_text),
            #     )

    def setup_queue(self):
        self.queue = multiprocessing.Queue(-1)

    def __exit__(self):
        super().__exit__()
        self.queue.put_nowait(None)


def setup_log():
    logger = setup_logger(LOG_NAME)
    # set roll file
    # rf = RFHandler(LOG_PATH, maxBytes=1024*1024*100, backupCount=3, delay=0.05)
    # rf.setLevel(logging.INFO)
    # rf.setFormatter(formatter)

    # log
    logger.setLevel(logging.ERROR)
    logger.addHandler(MultiProcessLogHandler())
    # log = logging.getLogger()
    # log.setLevel(logging.INFO)
    # log.addHandler(rf)

    return logger


logger = setup_log()
# logger.propagate = False
