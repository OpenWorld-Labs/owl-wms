from physicsnemo.utils.profiling import Profiler
import torch

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from owl_wms.utils import load_from_config
from owl_wms.utils.owl_vae_bridge import get_decoder_only

from .profiler import profile_fn, print_results

allow_ops_in_compiled_graph()


def model_setup():
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

    return world_model, img_dec, audio_dec


def create_dummy_inputs():
    ## Dummy Inputs
    dummy_x = torch.randn(1, 1, 128, 4, 4).bfloat16().cuda()
    dummy_audio = torch.randn(1, 1, 64).bfloat16().cuda()
    ts = torch.ones(1,1).bfloat16().cuda()
    mouse = torch.randn(1,1,2).bfloat16().cuda()
    btn = torch.randint(0, 1, (1,1,11)).bfloat16().cuda()

    dummy = (dummy_x, dummy_audio, ts, mouse, btn)
    dummy_pred_audio = torch.randn(1, 64, 120).bfloat16().cuda()

    return dummy, dummy_pred_audio


def profile_baseline(world_model, img_dec, audio_dec, dummy, dummy_pred_audio):
    ## Baseline Profile
    torch._dynamo.reset()

    res_wm = profile_fn(world_model, dummy)
    print_results(res_wm, "Baseline - WM")

    res_img = profile_fn(img_dec, dummy[0][0])
    print_results(res_img, "Baseline - IMG")

    res_audio = profile_fn(audio_dec, dummy_pred_audio)
    print_results(res_audio, "Baseline - AUDIO")


if __name__ == "__main__":
    # PhysicsNemo Profiler which is a singleton class so can set the configs here
    profiler = Profiler()
    profiler.enable("torch")

    world_model, img_dec, audio_dec = model_setup()
    dummy, dummy_pred_audio = create_dummy_inputs()

    profile_baseline(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)

    from .inductor_compile import profile_torch_compile_inductor, profile_torch_compile_inductor_fp8_torchao
    print("-------------------------------- Inductor Compile --------------------------------")
    torch._dynamo.reset()
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.epilogue_fusion = False
    # torch._inductor.config.coordinate_descent_tuning = True
    # torch._inductor.config.coordinate_descent_check_all_directions = True
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    torch.backends.cudnn.benchmark = True
    # torch._inductor.config.force_fuse_int_mm_with_mul = True
    # torch._inductor.config.use_mixed_mm = True
    try:
        # profile_torch_compile_inductor(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)

        profile_torch_compile_inductor_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)
    except Exception as e:
        print(f"Error in inductor compile: {e}")


    # from .tensorrt_compile import profile_torch_compile_tensorrt, profile_torch_compile_tensorrt_fp8_tensorrt
    # print("-------------------------------- TensorRT Compile --------------------------------")
    # torch._dynamo.reset()
    # try:
    #     profile_torch_compile_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)

    #     profile_torch_compile_tensorrt_fp8_torchao(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)

    #     profile_torch_compile_tensorrt_fp8_tensorrt(world_model, img_dec, audio_dec, dummy, dummy_pred_audio)
    # except Exception as e:
    #     print(f"Error in tensorrt compile: {type(e)}")
    #     print(f"{e} {e.__traceback__}")

    profiler.finalize()
