# owl_wms
Basic world models

## Docker setup:
```
docker build -t owl_wms .

docker run --gpus all -it \
  -v $HOME/.gitconfig:/root/.gitconfig:ro \
  -v $HOME/.ssh:/root/.ssh:ro \
  --shm-size 1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  owl_wms /bin/bash
```
