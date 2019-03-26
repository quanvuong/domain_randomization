# Domain Randomization

### Code Instructions

To run the experiment on the Walker environment with initial seed 0, please do:

```bash
$ docker run -v  $(pwd):/root/work/domain-randomization -v <absolute path to .mujoco folder>:/root/.mujoco -it sharadmv/domain-randomization:pytorch mpirun -np 30 pipenv run python scripts/ppo_pytorch.py --env_dist_stdev 1.0 --seed 0
```

During training, you can view the hyper-parameters of the run and training progress by opening up tensorboardX:

```bash
$ tensorboard --logdir runs
```
