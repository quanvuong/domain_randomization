#!/usr/bin/env bash

for ((i=15;i<30;i+=1))
do
    docker run --detach \
     -v $(pwd):/root/work/domain-randomization -it sharadmv/domain-randomization \
     pipenv run python scripts/ppo.py  --seed $i
done
