import torch
import time
import numpy as np

def time_with_cuda_events(func):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    func()
    end_event.record()
    
    # Wait for GPU to finish
    end_event.synchronize()
    
    # Get time in milliseconds
    return start_event.elapsed_time(end_event)

@torch.no_grad()
def time_fn(fn, dummy_input, n_warmup=10, n_eval=10):
    def x():
        if isinstance(dummy_input, tuple):
            return tuple(torch.randn_like(t) for t in dummy_input)
        return torch.randn_like(dummy_input)
    
    def wrapper():
        inputs = x()
        if isinstance(inputs, tuple):
            _ = fn(*inputs)
        else:
            _ = fn(inputs)

    for _ in range(n_warmup):
        wrapper()
    
    times = []
    for _ in range(n_eval):
        times.append(time_with_cuda_events(wrapper))
    times = np.array(times)

    return {
        'mean': np.mean(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def get_fps(t):
    return 1. / t