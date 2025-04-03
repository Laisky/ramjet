# from ramjet.engines import ioloop, thread_executor
# from ramjet.utils.log import logger as ramjet_logger

# from .notes import run as run_notes

# logger = ramjet_logger.getChild("tasks.telegram.notes")


# def bind_task():
#     def run():
#         logger.info("running...")
#         thread_executor.submit(run_notes)
#         ioloop.call_later(10 * 60, run)  # every 10 minutes

#     run()
