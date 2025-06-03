import torch
from torch import nn
import torch.nn.functional as F

class SimpleSampler:
    def __init__(self, n_steps=64):
        self.n_steps = n_steps

    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn = None, scale = 1):
        sampling_steps = self.n_steps
        x = torch.randn_like(dummy_batch)
        ts = torch.ones(x.shape[0], device=x.device,dtype=x.dtype)
        dt = 1. / sampling_steps

        for _ in range(sampling_steps):
            pred = model(x, ts, mouse, btn)
            x = x - pred*dt
            ts = ts - dt

        if decode_fn is not None:
            x = x * scale
            x = decode_fn(x)
        return x, mouse, btn

class InpaintSimpleSampler:
    def __init__(self, n_steps=64):
        self.n_steps = n_steps

    @torch.no_grad()
    def __call__(self, model, dummy_batch, mouse, btn, decode_fn = None, scale = 1):
        sampling_steps = self.n_steps

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
            x = x * scale
            x = decode_fn(x)
        return x, mouse, btn


if __name__ == "__main__":
    model = lambda x,t,m,b: x

    sampler = Sampler()
    x = sampler(model, torch.randn(4, 3, 64, 64), 
                torch.randn(4, 2), torch.randn(4, 8))
    print(x.shape)