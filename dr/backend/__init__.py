from dr.backend.dart import DartBackend
from dr.backend.mujoco import MujocoBackend

BACKENDS = {
    'dart': DartBackend(),
    'mujoco': MujocoBackend(),
}


def get_backend(backend_name):
    if backend_name not in BACKENDS:
        raise Exception(f"Backend {backend_name} not found")
    return BACKENDS[backend_name]
