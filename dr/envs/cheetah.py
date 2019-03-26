import numpy as np

from dr.envs.driver import Driver


class Cheetah(Driver):

    def get_parameters(self, env):
        return {
            'mass': self.backend.get_masses(env),
            'damping': self.backend.get_damping_coefficients(env),
            'gravity': self.backend.get_gravity(env),
        }
        return np.concatenate([self.backend.get_masses(env)[:], self.backend.get_damping_coefficients(env),
                               [self.backend.get_gravity(env)]])

    def set_parameters(self, env, parameters):
        masses = parameters['mass']
        gravity = parameters['gravity']
        damping = parameters['damping']

        assert np.all(masses > 0), 'masses can not be 0 or negative'

        self.backend.set_masses(env, masses)
        self.backend.set_gravity(env, gravity)
        self.backend.set_damping_coefficients(env, damping)
