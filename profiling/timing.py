import torch
import time
import numpy as np


@torch.inference_mode()
def time_with_cuda_events(func):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    func()
    end_event.record()
    
    # Wait for GPU to finish
    end_event.synchronize()
    
    # Get time in milliseconds, memory in MB
    return start_event.elapsed_time(end_event), torch.cuda.max_memory_allocated()/1024**2


@torch.inference_mode()
def time_fn(fn, dummy_input, n_warmup=10, n_eval=10):
    def x():
        if isinstance(dummy_input, tuple):
            return tuple(torch.randn_like(t) for t in dummy_input)
        return torch.randn_like(dummy_input)

    inputs = x()
    def wrapper(inputs):
        if isinstance(inputs, tuple):
            _ = fn(*inputs)
        else:
            _ = fn(inputs)

    for _ in range(n_warmup):
        wrapper(inputs)

    times = []
    memories = []
    for _ in range(n_eval):
        time, memory = time_with_cuda_events(lambda: wrapper(inputs))
        times.append(time)
        memories.append(memory)
    times = np.array(times)
    memories = np.array(memories)

    return {
        'mean_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'mean_memory': np.mean(memories),
        'min_memory': np.min(memories),
        'max_memory': np.max(memories),
        'std_memory': np.std(memories)
    }

def get_fps(t):
    return 1. / t