import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

from termcolor import colored

from owl_wms.models import get_model_cls
from owl_wms.configs import Config as RunConfig
from owl_wms.utils.owl_vae_bridge import get_decoder_only
from owl_wms.utils import freeze

@dataclass(frozen=True)
class ModelPaths:
    config: Path
    checkpoint: Path
    
    @classmethod
    def from_strings(cls, config_path: str, checkpoint_path: str) -> 'ModelPaths':
        return cls(
            config=Path(config_path),
            checkpoint=Path(checkpoint_path)
        )
    
    def validate(self) -> None:
        if not self.config.exists():
            raise FileNotFoundError(f"Config file not found: {self.config}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint}")


class ModelLoader:    
    DEFAULT_PATHS = ModelPaths.from_strings(
        config_path='/home/sami/owl-wms/checkpoints/wm/dcae_hf_cod/basic.yml',
        checkpoint_path='/home/sami/owl-wms/checkpoints/wm/dcae_hf_cod/ckpt_165k_ema.pt'
    )
    
    def __init__(self, paths: Optional[ModelPaths] = None):
        self.paths = paths or self.DEFAULT_PATHS
        self.paths.validate()
    
    @staticmethod
    def _append_state_dict_prefix(state_dict: Dict, prefix: str = 'core.') -> Dict:
        return {prefix+key: value for key, value in state_dict.items()}
    
    @staticmethod
    def _count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
    
    def _load_config(self) -> RunConfig:
        return RunConfig.from_yaml(str(self.paths.config))
    
    def _load_checkpoint(self) -> Dict:
        return torch.load(str(self.paths.checkpoint), map_location='cpu')
    
    def load_model(self, 
                   device: Optional[Union[str, torch.device]] = None,
                   eval_mode: bool = True,
                   verbose: bool = True) -> nn.Module:
        # Load configuration and create model
        config = self._load_config()
        model_cls = get_model_cls(config.model.model_id)
        model = model_cls(config.model)
        
        # Load and filter state dict
        checkpoint = self._load_checkpoint()
        
        if config.model.model_id == "game_rft":
            checkpoint = self._append_state_dict_prefix(checkpoint)

        model.load_state_dict(checkpoint)
        
        # Configure model
        if eval_mode:
            model.eval()
        
        if device is not None:
            model = model.to(device)
        
        # Print model information if requested
        if verbose:
            param_count = self._count_parameters(model)
            print(f'{colored("Model loaded", "blue")}\t\t {colored("successfully", "green")}')
            print(f'{colored("Parameters", "blue")}  \t\t {colored(f"{param_count:,}", "green")}')
            print(f'{colored("Config", "blue")}      \t\t {colored(str(self.paths.config), "green", attrs=["bold"])}')
            print(f'{colored("Checkpoint", "blue")}  \t\t {colored(str(self.paths.checkpoint), "green", attrs=["bold"])}')
        
        return model
    
    def load_decoder(self,
                     device: Optional[Union[str, torch.device]] = None,
                     eval_mode: bool = True,
                     verbose: bool = True) -> nn.Module:
        decoder = get_decoder_only()
        freeze(decoder)
        
        if verbose:
            print(f'{colored("Decoder loaded", "blue")}\t\t {colored("successfully", "green")}')
            print(f'{colored("Parameters", "blue")}  \t\t {colored(f"{self._count_parameters(decoder):,}", "green")}')
            print(f'{colored("Config", "blue")}      \t\t {colored(str(self.paths.config), "green", attrs=["bold"])}')
            print(f'{colored("Checkpoint", "blue")}  \t\t {colored(str(self.paths.checkpoint), "green", attrs=["bold"])}')
        
        if device is not None:
            decoder = decoder.to(device)
        
        if eval_mode:
            decoder.eval()

        return decoder


def load_models(config_path: Optional[str] = None,
                checkpoint_path: Optional[str] = None,
                device: Optional[Union[str, torch.device]] = None,
                eval_mode: bool = True,
                verbose: bool = True) -> tuple[nn.Module, nn.Module, RunConfig]:
    """
    Convenience function for loading models with custom paths.
    
    Args:
        config_path: Path to model configuration file
        checkpoint_path: Path to model checkpoint file
        device: Target device for the model
        eval_mode: Whether to set model to evaluation mode
        verbose: Whether to print model information
        
    Returns:
        Loaded PyTorch model
    """
    if config_path or checkpoint_path:
        # Use custom paths if provided
        paths = ModelPaths.from_strings(
            config_path or str(ModelLoader.DEFAULT_PATHS.config),
            checkpoint_path or str(ModelLoader.DEFAULT_PATHS.checkpoint)
        )
        loader = ModelLoader(paths)
    else:
        # Use default paths
        loader = ModelLoader()
    
    encoder = loader.load_model(device=device, eval_mode=eval_mode, verbose=verbose)
    decoder = loader.load_decoder(device=device, eval_mode=eval_mode, verbose=verbose)
    return encoder, decoder, loader._load_config()


if __name__ == "__main__":
    # Example usage with different approaches
    
    # Method 1: Using the class directly
    print("=== Loading model using ModelLoader class ===")
    loader = ModelLoader()
    model = loader.load_model()
    print(f"Model type: {type(model).__name__}")

    # Method 2: Using the convenience function
    print("=== Loading model using convenience function ===")
    encoder, decoder, config = load_models()
    print(f"Config: {config}")
