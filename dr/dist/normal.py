import numpy as np
import scipy.stats as stats

from dr.dist.env_dist import EnvironmentDistribution


class Normal(EnvironmentDistribution):

    def __init__(self, *args, **kwargs):
        self.scale = kwargs.pop('stdev', None)
        self.stdev_dict = kwargs.pop('stdev_dict', None)

        assert self.scale is None or self.stdev_dict is None

        super(Normal, self).__init__(*args, **kwargs)

    def _sample(self, params, mode):

        if mode == 'scale':
            return self._sample_with_scale(params)
        elif mode == 'dict':
            return self._sample_with_dict(params)
        else:
            exit('What went wrong here?')

    def _sample_with_scale(self, params):
        if self.scale == 0.0:
            return params.copy()

        # Sample gravity parameter
        gravity = np.random.normal(params['gravity'], scale=self.scale)

        # Mass parameters can only be positive,
        # We thus sample them from a truncated normal distribution
        # We set the lower bound to an arbitrary small positive number
        # since stats.truncnorm samples in closed interval [a, b].
        # The distribution is theoretically un-bounded from above
        # But we set the upper bound to be an arbitrary large number
        # to avoid any potential numerical issue with setting upper bound to be +inf
        mass_means = params['mass']
        low, high = (1e-7 - mass_means) / self.scale, (1e7 - mass_means) / self.scale
        mass = stats.truncnorm(low, high, loc=mass_means, scale=self.scale).rvs()

        damping_means = params['damping']
        low, high = (0 - damping_means) / self.scale, (1e7 - damping_means) / self.scale
        damping = stats.truncnorm(low, high, loc=damping_means, scale=self.scale).rvs()

        sample = {
            'mass': mass,
            'damping': damping,
            'gravity': gravity,
        }

        assert len(sample) == len(params)
        return sample

    def _sample_with_dict(self, params):

        # Sample gravity parameter
        gravity = np.random.normal(params['gravity'], scale=self.stdev_dict['gravity'])

        # Mass parameters can only be positive,
        # We thus sample them from a truncated normal distribution
        # We set the lower bound to an arbitrary small positive number
        # since stats.truncnorm samples in closed interval [a, b].
        # The distribution is theoretically un-bounded from above
        # But we set the upper bound to be an arbitrary large number
        # to avoid any potential numerical issue with setting upper bound to be +inf
        mass_means = params['mass']
        mass_stdev = self.stdev_dict['mass']

        low, high = (1e-7 - mass_means) / mass_stdev, (1e7 - mass_means) / mass_stdev
        mass = stats.truncnorm(low, high, loc=mass_means, scale=mass_stdev).rvs()

        damping_means = params['damping']
        damping_stdev = self.stdev_dict['damping']

        low, high = (0 - damping_means) / damping_stdev, (1e7 - damping_means) / damping_stdev
        damping = stats.truncnorm(low, high, loc=damping_means, scale=damping_stdev).rvs()

        sample = {
            'mass': mass,
            'damping': damping,
            'gravity': gravity,
        }

        assert len(sample) == len(params)
        return sample
