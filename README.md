# owl_wms
Basic world models

## Docker setup:
```
docker build -t owl_wms .

docker run --gpus all -it \
  -v $HOME/.gitconfig:/root/.gitconfig:ro \
  -v $HOME/.ssh:/root/.ssh:ro \
  owl_wms /bin/bash
```
