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
    

