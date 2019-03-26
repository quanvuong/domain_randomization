from abc import ABCMeta, abstractmethod


class Driver(object, metaclass=ABCMeta):

    def __init__(self, backend):
        self.backend = backend

    @abstractmethod
    def get_parameters(self, env):
        pass

    @abstractmethod
    def set_parameters(self, env, parameters):
        pass
