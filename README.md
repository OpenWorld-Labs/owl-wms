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

## Sample Model Checkpoints:
audio vae:
```
wget https://model-checkpoints.fly.storage.tigris.dev/cod_audio_20k_ema.pt -O checkpoints/owl_vaes/cod_audio_20k_ema.pt
```
image vae:
```
wget https://model-checkpoints.fly.storage.tigris.dev/cod_128x_30k_ema.pt -O checkpoints/owl_vaes/cod_128x_30k_ema.pt
```
200m av wm:
```
wget https://model-checkpoints.fly.storage.tigris.dev/av_dfot_85k_ema_200m.pt -O checkpoints/av_huge/av_dfot_85k_ema_200m.pt
```

### Reference docs for FP8:
1. https://github.com/pytorch/ao
2. https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/flux/README.md
3. https://github.com/xdit-project/xDiT/tree/main
4. https://github.com/facebookresearch/DiT/tree/main