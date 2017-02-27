import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ramjet.settings import N_PROCESS_WORKER, N_THREAD_WORKER


thread_executor = ThreadPoolExecutor(max_workers=N_THREAD_WORKER)
process_executor = ProcessPoolExecutor(max_workers=N_PROCESS_WORKER)
ioloop = asyncio.get_event_loop()
