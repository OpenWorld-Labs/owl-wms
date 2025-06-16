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

## Optimization history:
**1B Model notes (for 1 step with KV cache)**:
1. Compile:
Mean: 85.35ms | 10 FPS
Min: 78.28ms
Max: 93.37ms

World model parameters: 1,093,823,296    
Image decoder parameters: 73,355,776    
Audio decoder parameters: 20,950,528    

*  Baseline - WM    
Mean: 29.00ms, 2291.17MB    
Min: 28.68ms, 2291.17MB    
Max: 29.73ms, 2291.17MB    
Std: 0.31ms, 0.00MB    
Avg FPS: 34.48FPS

*  Baseline - IMG    
Mean: 28.08ms, 3027.23MB    
Min: 27.97ms, 3025.68MB    
Max: 28.91ms, 3027.68MB    
Std: 0.27ms, 0.57MB    
Avg FPS: 35.61FPS

*  Baseline - AUDIO    
Mean: 12.52ms, 2564.69MB    
Min: 12.44ms, 2564.39MB    
Max: 12.64ms, 2564.73MB    
Std: 0.05ms, 0.10MB    
Avg FPS: 79.90FPS



*  Torch Compile - WM    
Mean: 3.10ms, 2283.84MB    
Min: 3.07ms, 2283.84MB    
Max: 3.14ms, 2283.84MB    
Std: 0.02ms, 0.00MB    
Avg FPS: 322.44FPS


*  Torch Compile - IMG    
Mean: 21.52ms, 2283.83MB    
Min: 21.49ms, 2283.83MB    
Max: 21.67ms, 2283.83MB    
Std: 0.05ms, 0.00MB    
Avg FPS: 46.47FPS

*  Torch Compile - AUDIO    
Mean: 3.70ms, 2281.84MB    
Min: 3.69ms, 2281.84MB    
Max: 3.71ms, 2281.84MB    
Std: 0.01ms, 0.00MB    
Avg FPS: 270.51FPS
