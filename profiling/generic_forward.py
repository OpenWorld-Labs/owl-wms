import torch
from physicsnemo.utils.profiling import Profiler
import torch_tensorrt

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results


# PhysicsNemo Profiler which is a singleton class so can set the configs here
profiler = Profiler()
profiler.enable("torch")


def reset_torch_compiler_configs():
    torch._dynamo.reset()
    ## Torch Dynamo Setup
    allow_ops_in_compiled_graph()
    # torch.compile flags
    torch._inductor.config.conv_1x1_as_mm = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.benchmark = True

    torch._inductor.config.max_autotune = True  # redundant with mode='max-autotune'
    torch._inductor.config.max_autotune_gemm = True  # not redundant, by default False
    torch._inductor.config.max_autotune_pointwise = True  # not redundant, by default False


## Model Setup
wm_cfg = "configs/av.yml"
vae_cfg = "configs/owl_vaes/cod_128x.yml"
audio_vae_cfg = "configs/owl_vaes/cod_audio.yml"

world_model = load_from_config(wm_cfg).core.bfloat16().cuda().eval()
img_dec = get_decoder_only(None, vae_cfg).decoder.bfloat16().cuda().eval()
audio_dec = get_decoder_only(None, audio_vae_cfg).decoder.bfloat16().cuda().eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"World model parameters: {count_parameters(world_model):,}    ")
print(f"Image decoder parameters: {count_parameters(img_dec):,}    ")
print(f"Audio decoder parameters: {count_parameters(audio_dec):,}    ")
print()


## Dummy Inputs
dummy_x = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
dummy_audio = torch.randn(1, 1, 64).bfloat16().cuda()
ts = torch.ones(1,1).bfloat16().cuda()
mouse = torch.randn(1,1,2).bfloat16().cuda()
btn = torch.randint(0, 1, (1,1,11)).bfloat16().cuda()

dummy = (dummy_x, dummy_audio, ts, mouse, btn)
dummy_pred_audio = torch.randn(1, 64, 120).bfloat16().cuda()

## Baseline Profile
reset_torch_compiler_configs()

res_wm = profile_fn(world_model, dummy)
print_results(res_wm, "Baseline - WM")

res_img = profile_fn(img_dec, dummy_x[0])
print_results(res_img, "Baseline - IMG")

res_audio = profile_fn(audio_dec, dummy_pred_audio)
print_results(res_audio, "Baseline - AUDIO")

# ## Torch Compile with Inductor
# reset_torch_compiler_configs()

# compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
# compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
# compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

# res_wm = profile_fn(compiled_world_model, dummy)
# print_results(res_wm, "Torch Compile - WM")

# res_img = profile_fn(compiled_img_dec, dummy_x[0])
# print_results(res_img, "Torch Compile - IMG")

# res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
# print_results(res_audio, "Torch Compile - AUDIO")

# ## Torch Compile with Torch-TensorRT
reset_torch_compiler_configs()

tensorrt_backend_args = {
    "use_explicit_typing": True,
    "enabled_precisions": {torch.float32, torch.bfloat16, torch.float16},
    'max_autotune': True,
    'triton.cudagraphs': True,
    'coordinate_descent_tuning': True,
    'debug': True,
}

compiled_world_model = torch.compile(world_model, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=tensorrt_backend_args)

# imge_cnn_configs = {
#     "device": torch_tensorrt.Device("dla:0", allow_gpu_fallback=True)
#     } | tensorrt_backend_args
# compiled_img_dec = torch.compile(img_dec, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=imge_cnn_configs)

# compiled_audio_dec = torch.compile(audio_dec, backend="torch_tensorrt", dynamic=False, fullgraph=True, options=tensorrt_backend_args)

res_wm = profile_fn(compiled_world_model, dummy)
print_results(res_wm, "Torch Compile TensorRT - WM")

# res_img = profile_fn(compiled_img_dec, dummy_x[0])
# print_results(res_img, "Torch Compile TensorRT - IMG")

# res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
# print_results(res_audio, "Torch Compile TensorRT - AUDIO")

# ## Torch Compile with Torch-TensorRT and OneDiff

# ## Torch Compile + FP8
# dummy = (dummy_x, dummy_audio, ts, mouse, btn)
# dummy_pred_audio = torch.randn(1, 64, 120).bfloat16().cuda()

# ## Torch Compile + FP4


profiler.finalize()