import os.path as osp
from datetime import datetime

import click
from git import Repo
from mpi4py import MPI

import dr

COMM = MPI.COMM_WORLD

repo = Repo('./')
branch = repo.active_branch.name


@click.command()
@click.option('--experiment_name', type=str, default=osp.join(branch, datetime.now().strftime('%b%d_%H-%M-%S')))
@click.option('--env_name', type=str, default='Walker')
@click.option('--backend', type=str, default='dart')
@click.option('--collision_detector', type=str, default='bullet')
@click.option('--num_timesteps', type=int, default=30000)
@click.option('--seed', type=int, default=0)
@click.option('--env_dist_stdev', type=float, default=0.0)
@click.option('--mean_scale', type=float, default=1.0)
@click.option('--pop_size', type=int, default=30)
@click.option('--num_elites', type=int, default=10)
@click.option('--debug', type=bool, default=False)
@click.option('--num_eval_env', type=int, default=100)
def main(experiment_name, env_name, backend, collision_detector,
         num_timesteps, seed, env_dist_stdev, mean_scale,
         pop_size, num_elites,
         debug, num_eval_env):
    assert env_dist_stdev == 1.0
    assert mean_scale == 1.0
    assert env_name == 'Walker'

    if debug:
        pop_size = 3
        num_elites = 2
        num_timesteps = 20000
        num_eval_env = 10

    assert pop_size == COMM.Get_size()

    dr.experiment.PPO_Pytorch(
        experiment_name,
        env_params=dict(
            env_name=env_name,
            backend=backend,
            collision_detector=collision_detector
        ),
        train_params=dict(
            num_timesteps=num_timesteps,
            seed=seed,
            env_dist_stdev=env_dist_stdev,
            mean_scale=mean_scale,

            # PPO parameter,
            hid_size=64,
            pol_init_std=1.0,
            adam_epsilon=1e-5,
            optim_stepsize=3e-4,

            ts_per_batch=2048,
            lam=0.95,
            gamma=0.99,
            optim_epoch=10,
            optim_batch_size=64,
            clip_param=0.2,

            # CEM parameters
            pop_size=pop_size,
            num_elites=num_elites,
            num_eval_env=num_eval_env
        )
    ).run()


if __name__ == "__main__":
    main()
