#!/usr/bin/env bash

for ((i=0;i<5;i+=1))
do
    docker run --detach -v  $(pwd):/root/work/domain-randomization -v /home/qvuong/.mujoco:/root/.mujoco -it sharadmv/domain-randomization:pytorch mpirun -np 30 pipenv run python scripts/ppo_pytorch.py --env_dist_stdev 1.0 --seed $i
done

