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

### Reference docs for FP8:
1. https://github.com/pytorch/ao
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/flux/README.md
3. https://github.com/xdit-project/xDiT/tree/main
4. https://github.com/facebookresearch/DiT/tree/main