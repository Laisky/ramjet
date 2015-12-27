from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import tornado.ioloop

from ramjet.settings import N_THREAD_WORKER, N_PROCESS_WORKER
# ---------------- IMPORT YOUR TASKS 👇👇👇 ----------------
from .heart import bind_task as bind_heart_task
from .snmp import bind_task as bind_snmp_task
from .sync_projects import bind_task as bind_projects_task


thread_executor = ThreadPoolExecutor(max_workers=N_THREAD_WORKER)
process_executor = ProcessPoolExecutor(max_workers=N_PROCESS_WORKER)


def setup_tasks(ioloop=False):
    ioloop = ioloop or tornado.ioloop.IOLoop.instance()
    # Add your tasks below!
    # your_task(ioloop, thread_executor, process_executor)
    # ---------------- YOUR TASKS 👇👇👇 ----------------
    bind_heart_task(ioloop, thread_executor, process_executor)
    bind_snmp_task(ioloop, thread_executor, process_executor)
    bind_projects_task(ioloop, thread_executor, process_executor)
