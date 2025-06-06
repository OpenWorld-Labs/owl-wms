from __future__ import annotations

import os
import yaml
from typing import Optional
from dataclasses import dataclass

from owl_wms.configs import Config as RunConfig, TransformerConfig as ModelConfig

@dataclass
class WebappConfig:
    model_checkpoint_path : os.PathLike
    model_config_path : os.PathLike
    model_config : ModelConfig
    run_config : RunConfig
    stream_config : StreamingConfig
    sampling_config : SamplingConfig
    device : str = 'cuda'

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> WebappConfig:
        
        with (open(path, 'r')                           as wcp,
              open(config['model_config_path'], 'r')    as mcp,
              open(config['run_config_path'], 'r')      as rcp):
            config = yaml.safe_load(wcp)
            config['model_config'] = yaml.safe_load(mcp)
            config['run_config'] = yaml.safe_load(rcp)

        return cls(**config)


@dataclass    
class SamplingConfig:
    sampling_steps : int = 20
    cfg_scale : float = 1.3
    num_frames : int = 60
    noise_prev : float = 0.2
    window_length : Optional[int] = 60

@dataclass
class StreamingConfig:
    fps: int = 20
    frames_per_batch: int = 8
    window_length: int = 60
    device: str = 'cuda'
    n_buttons: int = 11
    mouse_range: tuple[float, float] = (-1.0, 1.0)

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.fps

    @property
    def batch_duration(self) -> float:
        return self.frames_per_batch / self.fps
