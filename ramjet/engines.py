import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ramjet.settings import N_PROCESS_WORKER, N_THREAD_WORKER


class LazyExecutor:
    def __init__(self, executor_class, *args, **kw):
        self.executor = None
        self.executor_class = executor_class
        self.args = args
        self.kw = kw

    def setup_executor(self):
        self.executor = self.executor_class(*self.args, **self.kw)

    def __getattr__(self, name):
        if not self.executor:
            self.setup_executor()

        return getattr(self.executor, name)


thread_executor: ThreadPoolExecutor = LazyExecutor(
    ThreadPoolExecutor, max_workers=N_THREAD_WORKER
)
process_executor: ProcessPoolExecutor = LazyExecutor(
    ProcessPoolExecutor, max_workers=N_PROCESS_WORKER
)
ioloop = asyncio.get_event_loop()


def shutdown_all_engines():
    if ioloop.is_running():
        ioloop.stop()

    thread_executor.shutdown(wait=False)
    process_executor.shutdown(wait=False)
