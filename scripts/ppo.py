import os.path as osp
from datetime import datetime

import click
from git import Repo

import dr

repo = Repo('./')
branch = repo.active_branch.name


@click.command()
@click.option('--experiment_name', type=str, default=osp.join(branch, datetime.now().strftime('%b%d_%H-%M-%S')))
@click.option('--env_name', type=str, default='Hopper')
@click.option('--backend', type=str, default='dart')
@click.option('--collision_detector', type=str, default='bullet')
@click.option('--num_timesteps', type=int, default=1e6)
@click.option('--seed', type=int, default=0)
@click.option('--env_dist_stdev', type=float, default=0.0)
@click.option('--mean_scale', type=float, default=1.0)
def main(experiment_name, env_name, backend, collision_detector,
         num_timesteps, seed, env_dist_stdev, mean_scale):
    assert env_dist_stdev == 0.0
    assert mean_scale == 1.0

    dr.experiment.PPO(
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
            mean_scale=mean_scale
        )
    ).run()


if __name__ == "__main__":
    main()
