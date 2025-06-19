from physicsnemo.utils.profiling import Profiler
import torch
import modelopt.torch.quantization as mtq  # torch-tensorrt model optimizer
import torch_tensorrt as torchtrt
from modelopt.torch.quantization.utils import export_torch_mode

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results


def profile_torch_compile_inductor(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor
    reset_torch_compiler_configs()

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
    pass


def profile_torch_compile_inductor_fp8_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor + FP8 with torch-tensorrt
    pass
