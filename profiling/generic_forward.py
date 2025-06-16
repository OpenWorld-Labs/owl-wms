import torch

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from profiling.timing import time_fn

import torch._dynamo
torch._dynamo.config.debug = True

wm_cfg = "configs/av.yml"
vae_cfg = "configs/owl_vaes/cod_128x.yml"
audio_vae_cfg = "configs/owl_vaes/cod_audio.yml"

world_model = load_from_config(wm_cfg).core.bfloat16().cuda()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"World model parameters: {count_parameters(world_model):,}")

dummy_x = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
dummy_audio = torch.randn(1, 1, 64).bfloat16().cuda()
ts = torch.ones(1,1).bfloat16().cuda()
mouse = torch.randn(1,1,2).bfloat16().cuda()
btn = torch.randint(0, 1, (1,1,11)).bfloat16().cuda()

dummy = (dummy_x, dummy_audio, ts, mouse, btn)

torch.compile(world_model, dynamic = False, fullgraph=True)
res = time_fn(world_model, dummy)
print(f"Mean: {res['mean']:.2f}ms")
print(f"Min: {res['min']:.2f}ms")
print(f"Max: {res['max']:.2f}ms")
print(f"Avg FPS: {1000./res['mean']:.2f}FPS")

img_dec = get_decoder_only(None, vae_cfg)
audio_dec = get_decoder_only(None, audio_vae_cfg)

"""
1B Model notes (for 1 step with KV cache)
Mean: 85.35ms | 10 FPS
Min: 78.28ms
Max: 93.37ms

"""
