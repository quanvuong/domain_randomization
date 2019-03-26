# domain-randomization

# Docker instructions

```bash
$ docker pull sharadmv/domain-randomization
$ docker run --runtime=nvidia -v $(pwd):/root/work/domain-randomization -it sharadmv/domain-randomization pipenv run python scripts/ppo.py
```
