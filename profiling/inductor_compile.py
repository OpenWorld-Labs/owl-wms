from physicsnemo.utils.profiling import Profiler
import torch

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results

# model quantization imports
import modelopt.torch.quantization as mtq  # torch-tensorrt model optimizer
import torch_tensorrt as torchtrt
from modelopt.torch.quantization.utils import export_torch_mode


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
    pass


def profile_torch_compile_inductor_fp8_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Torch Compile with Inductor + FP8 with torch-tensorrt
    quant_cfg = mtq.FP8_DEFAULT_CFG

    def calibrate_loop(model):
        # add a training loop to calibrate the quantized model
        pass

    quantized_world_model = mtq.quantize(world_model, quant_cfg, forward_loop=calibrate_loop)
    # quantized_img_dec = mtq.quantize(img_dec, quant_cfg, forward_loop=calibrate_loop)
    # quantized_audio_dec = mtq.quantize(audio_dec, quant_cfg, forward_loop=calibrate_loop)

    compiled_world_model = quantized_world_model # torch.compile(quantized_world_model)
    # compiled_img_dec = quantized_img_dec # torch.compile(quantized_img_dec)
    # compiled_audio_dec = quantized_audio_dec # torch.compile(quantized_audio_dec)

    res_wm = profile_fn(compiled_world_model, dummy)
    print_results(res_wm, "Torch Compile + FP8 TensorRT - WM")

    # res_img = profile_fn(compiled_img_dec, dummy[0][0])
    # print_results(res_img, "Torch Compile + FP8 TensorRT - IMG")

    # res_audio = profile_fn(compiled_audio_dec, dummy_pred_audio)
    # print_results(res_audio, "Torch Compile + FP8 TensorRT - AUDIO")