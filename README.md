## Domain Randomization

# Docker instructions

```bash
$ docker run -v  $(pwd):/root/work/domain-randomization -v <absolute path to .mujoco folder>:/root/.mujoco -it sharadmv/domain-randomization:pytorch mpirun -np 30 pipenv run python scripts/ppo_pytorch.py --env_dist_stdev 1.0 --seed 0
```
