# Profiling

We are aiming to hit relatively high frame-rates on models. The purpose of this folder is for profiling tests.

## Example run:
python -m profiling.generic_foreward

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
5. https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

## Potential ideas:
1. Use NVIDIA TensorRT backend instead of torch inductor for speedups on NVIDIA GPUs.
https://docs.pytorch.org/TensorRT/user_guide/torch_compile.html