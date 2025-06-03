import torch.distributed as dist
import wandb
import torch

import einops as eo

import numpy as np
from .vis import draw_frames

class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out). 
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        
        self.data = {}
    
    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}
        
        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)
            
            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced
            
        self.data = {}
        return final

@torch.no_grad()
def to_wandb(x, batch_mouse, batch_btn, gather = False, max_samples = 8):
    # x is [b,n,c,h,w]
    x = x.clamp(-1, 1)
    x = x[:max_samples]
a

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    # Get labels on them
    x = draw_frames(x, batch_mouse, batch_btn) # -> [b,n,c,h,w] [0,255] uint8 np

    if max_samples == 8:
        x = eo.rearrange(x, '(r c) n c h w -> n c (r h) (c w)', r = 2, c = 4)

    return wandb.Video(x, format='gif',fps=60)
    