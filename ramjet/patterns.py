from abc import abstractmethod

from ramjet.settings import logger as base_logger


class BaseWorker(object):
    def __init__(self, logger=None):
        self.logger = logger or base_logger

    @abstractmethod
    def gen_docu(self):
        pass

    def filter(self, docu):
        return docu

    @abstractmethod
    def worker(self, docu):
        pass

    def close(self):
        pass

    def done(self):
        pass

    def run(self):
        self.logger.debug("run BaseChecker")
        for docu in self.gen_docu():
            self.logger.debug("check docu {}".format(docu))
            docu = self.filter(docu)
            if docu:
                self.worker(docu)

        self.logger.info("BaseChecker done")
        self.done()

    def __del__(self):
        self.close()
