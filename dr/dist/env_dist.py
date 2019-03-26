import copy
from abc import abstractmethod, ABCMeta

import numpy as np

from dr.backend import get_backend
from dr.envs import get_driver


class EnvironmentDistribution(object, metaclass=ABCMeta):

    def __init__(self, env_name, backend_name, **kwargs):
        self.env_name, self.backend_name = env_name, backend_name
        self.backend = get_backend(backend_name)
        self.env_driver = get_driver(env_name, self.backend)
        self.root_env = self.backend.make(self.env_name)
        self.root_env.env.disableViewer = False

        mean_dict = kwargs.pop('mean_dict', None)
        mean_scale = kwargs.pop('mean_scale', None)

        assert (mean_scale is not None and mean_dict is not None) is False, 'Can only specify one options'

        if mean_dict is None and mean_scale is None:
            self.default_parameters = copy.deepcopy(self.env_driver.get_parameters(self.root_env))

        elif mean_dict is not None:
            self.default_parameters = mean_dict

        elif mean_scale is not None:
            self.default_parameters = copy.deepcopy(self.env_driver.get_parameters(self.root_env))
            self._scale_mean_scalar(mean_scale)

        self._seed = None

    def sample(self, mode, in_place=True):
        if in_place:
            env = self.root_env
            parameters = self.default_parameters
        else:
            env = self.backend.make(self.env_name)
            env.env.disableViewer = False
            parameters = self.env_driver.get_parameters(env)

        parameters = self._sample(parameters, mode)
        self.env_driver.set_parameters(env, parameters)
        return env

    def _scale_mean(self, mean_scale):

        assert list(self.default_parameters.keys()) == ['mass', 'damping', 'gravity'], \
            'Not all parameters are being scaled'

        self.default_parameters['mass'] *= mean_scale
        self.default_parameters['damping'] *= mean_scale
        self.default_parameters['gravity'] *= mean_scale

    def _scale_mean_scalar(self, mean_scale):

        assert list(self.default_parameters.keys()) == ['mass', 'damping', 'gravity'], \
            'Not all parameters are being scaled'

        self.default_parameters['mass'] *= mean_scale
        self.default_parameters['damping'] *= mean_scale
        self.default_parameters['gravity'] *= mean_scale

    @abstractmethod
    def _sample(self, parameters, mode):
        pass

    def seed(self, seed):
        np.random.seed(seed)
        self._seed = seed
