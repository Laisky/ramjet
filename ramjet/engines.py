from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from ramjet.settings import N_THREAD_WORKER, N_PROCESS_WORKER


thread_executor = ThreadPoolExecutor(max_workers=N_THREAD_WORKER)
process_executor = ProcessPoolExecutor(max_workers=N_PROCESS_WORKER)
ioloop = asyncio.get_event_loop()
