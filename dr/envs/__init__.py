from dr.envs.cheetah import Cheetah
from dr.envs.hopper import Hopper
from dr.envs.walker import Walker

__all__ = ['get_driver']

DRIVERS = {
    'Hopper': Hopper,
    'Cheetah': Cheetah,
    'Walker': Walker,
}


def get_driver(env_name, backend):
    if env_name not in DRIVERS:
        raise Exception(f"Environment {env_name} not found")
    return DRIVERS[env_name](backend)
