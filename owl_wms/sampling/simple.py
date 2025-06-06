import torch
from torch import nn
import torch.nn.functional as F

class SimpleSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, sampling_steps = 64, decode_fn = None, scale = 1):
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        for _ in range(sampling_steps):
            pred = model(x, ts, mouse, btn)
            x = x - pred*dt
            ts = ts - dt

        if decode_fn is not None:
            pixels = decode_fn(x * scale)
        else:
            pixels = None
        return x, pixels

class InpaintSimpleSampler:
    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, sampling_steps = 64, decode_fn = None, scale = 1):
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        dt = 1. / sampling_steps
        
        # Calculate midpoint
        mid = x.shape[1] // 2
        x[:,:mid] = dummy_batch[:,:mid]
        
        for _ in range(sampling_steps):
            pred = model(x, ts, mouse, btn)
            
            # Only update second half
            x[:, mid:] = x[:, mid:] - pred[:, mid:]*dt
            ts[:, mid:] = ts[:, mid:] - dt

        if decode_fn is not None:
            pixels = decode_fn(x * scale)
        else:
            pixels = None
        return x, pixels
