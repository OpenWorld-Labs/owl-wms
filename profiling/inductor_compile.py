from physicsnemo.utils.profiling import Profiler
import torch

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results

# torchao quantization
import torchao
# autoquant uses int8 by default with additional support for int4, not FP8
from torchao.quantization import (
    quantize_,
    Float8WeightOnlyConfig,
    Float8StaticActivationFloat8WeightConfig,
    Float8DynamicActivationFloat8WeightConfig,
    PerTensor,
    PerRow,
)


def profile_torch_compile_inductor(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor

    compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile - WM")

    res_img = profile_fn(compiled_img_dec, dummy[0][0])
    print_results(res_img, "Torch Compile - IMG")

    res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    print_results(res_audio, "Torch Compile - AUDIO")


def profile_torch_compile_inductor_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor + FP8 with torchao

    compiled_world_model = torch.compile(world_model, mode='max-autotune', dynamic=False, fullgraph=True)
    # compiled_img_dec = torch.compile(img_dec, mode='max-autotune', dynamic=False, fullgraph=True)
    # compiled_audio_dec = torch.compile(audio_dec, mode='max-autotune', dynamic=False, fullgraph=True)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile - WM")

    quantize_(compiled_world_model, Float8WeightOnlyConfig())
    # quantize_(compiled_img_dec, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
    # quantize_(compiled_audio_dec, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile + FP8 TorchAO - WM")

    # res_img = profile_fn(compiled_img_dec, dummy[0][0])
    # print_results(res_img, "Torch Compile + FP8 TorchAO - IMG")

    # res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    # print_results(res_audio, "Torch Compile + FP8 TorchAO - AUDIO")