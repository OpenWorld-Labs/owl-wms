from torch import nn, Tensor
from typing import Literal, Callable
from functools import partial, cache
from multimethod import multimethod
from owl_wms.sampling.cfg import CFGSampler, InpaintCFGSampler, WindowCFGSampler
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn

SAMPLING_STEPS = 60
SCALE = 2.17
CFG_SCALE = 1.3

MouseData   = Tensor
ButtonData  = Tensor
VideoData   = Tensor
LatentData  = Tensor

@multimethod
def create_sampler(sampler_id: Literal['cfg'], encoder: nn.Module, decoder: nn.Module,
                   batch_size: int      = 8,
                   sampling_steps: int  = SAMPLING_STEPS,
                   cfg_scale: float     = CFG_SCALE,
                   scale: float         = SCALE) -> Callable[[MouseData, ButtonData],
                                                        tuple[LatentData, VideoData]]:
    @cache # simple singleton
    def _sampler(): return CFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=sampling_steps,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=scale,
        cfg_scale=cfg_scale,
        model=encoder
    )

@multimethod
def create_sampler(sampler_id: Literal['inpaint_cfg'], encoder: nn.Module, decoder: nn.Module,
                   batch_size: int      = 8,
                   sampling_steps: int  = SAMPLING_STEPS,
                   cfg_scale: float     = CFG_SCALE,
                   scale: float         = SCALE) -> Callable[[MouseData, ButtonData],
                                                        tuple[LatentData, VideoData]]:
    @cache
    def _sampler(): return InpaintCFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=sampling_steps,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=scale,
        cfg_scale=cfg_scale,
        model=encoder
    )

@multimethod
def create_sampler(sampler_id: Literal['window'], encoder: nn.Module, decoder: nn.Module,
                   batch_size: int      = 8,
                   sampling_steps: int  = SAMPLING_STEPS,
                   cfg_scale: float     = CFG_SCALE,
                   scale: float         = SCALE) -> Callable[[MouseData, ButtonData],
                                                        tuple[LatentData, VideoData]]:
    @cache
    def _sampler(): return WindowCFGSampler()

    return partial(
        _sampler().__call__,
        sampling_steps=sampling_steps,
        decode_fn=make_batched_decode_fn(decoder, batch_size=batch_size),
        scale=scale,
        cfg_scale=cfg_scale,
        model=encoder
    )
