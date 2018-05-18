from abc import ABCMeta, abstractmethod


class Trainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_training(self):
        pass

    @abstractmethod
    def do_training(self, sess, saver):
        pass