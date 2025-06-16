import torch
from torch import nn

import time

def freeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = True

class Timer:
    def reset(self):
        self.start_time = time.time()
    
    def hit(self):
        return time.time() - self.start_time

def versatile_load(path):
    ckpt = torch.load(path, map_location = 'cpu', weights_only=False)
    if not 'ema' in ckpt and not 'model' in ckpt:
        return ckpt
    elif 'ema' in ckpt:
        ckpt = ckpt['ema']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("ema_model.module."):
                ddp_ckpt = True
        if ddp_ckpt:
            prefix = 'ema_model.module.'
        else:
            prefix = 'ema_model.'
    elif 'model' in ckpt:
        ckpt = ckpt['model']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("module."):
                ddp_ckpt = True
        if ddp_ckpt:
            prefix = 'module.'
        else:
            prefix = None
    
    if prefix is None:
        return ckpt
    else:
        ckpt = {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}
    
    return ckpt

def find_unused_params(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} has no gradient")

def load_from_config(cfg_path, ckpt_path = None):
    from ..configs import Config
    from ..models import get_model_cls

    cfg = Config.from_yaml(cfg_path)
    if hasattr(cfg, 'model'):
        cfg = cfg.model
    
    model = get_model_cls(cfg.model_id)(cfg)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt_path)
    return model

@torch.no_grad()
def batch_permute(mouse, button, factor = 1):
    """
    mouse: [b,n,2]
    button: [b,n,n_button]

    Clones mouse/button, randomly permutes along first dim, then concatenates
    Used to increase effective size of inputs for sampling purposes
    """

    for _ in range(factor):
        mouse_clone = mouse.clone()
        button_clone = button.clone()

        inds = torch.randperm(mouse.size(0))
        mouse_clone = mouse_clone[inds]
        button_clone = button_clone[inds]

        mouse = torch.cat([mouse, mouse_clone], dim = 1)
        button = torch.cat([button, button_clone], dim = 1)

    return mouse, button

@torch.no_grad()
def batch_permute_to_length(mouse, button, length):
    """
    Calls batch_permute with a factor that ensures output length >= target length,
    then truncates to exact length needed.
    
    Args:
        mouse: [b,n,2] tensor
        button: [b,n,n_button] tensor 
        length: Target sequence length
    Returns:
        mouse, button tensors with sequence length = length
    """
    # Calculate how many times we need to double n to exceed length
    n = mouse.shape[1]
    factor = 0
    doubled_length = n
    while doubled_length < length:
        factor += 1
        doubled_length *= 2
        
    # Do the permutation and truncate to exact size needed
    mouse, button = batch_permute(mouse, button, factor=factor)
    mouse = mouse[:,:length]
    button = button[:,:length]
    
    return mouse, button
    

