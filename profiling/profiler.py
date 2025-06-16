"""
This model is a simple implementation of torch.utils.benchmark.
"""

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
    output = func()
    end_event.record()
    
    # Wait for GPU to finish
    end_event.synchronize()
    
    # Get time in milliseconds, memory in MB
    return start_event.elapsed_time(end_event), torch.cuda.max_memory_allocated()/1024**2, output


@torch.inference_mode()
def profile_fn(fn, dummy_input, n_warmup=10, n_eval=10):
    def x():
        if isinstance(dummy_input, tuple):
            return tuple(torch.randn_like(t) for t in dummy_input)
        return torch.randn_like(dummy_input)

    inputs = x()
    def wrapper(inputs):
        if isinstance(inputs, tuple):
            output = fn(*inputs)
        else:
            output = fn(inputs)
        return output

    times = []
    memories = []
    with torch.profiler.profile() as prof:
        for _ in range(n_warmup):
            with torch.profiler.record_function('warmup'):
                wrapper(inputs)
            prof.step()

        for _ in range(n_eval):
            with torch.profiler.record_function('real execution'):
                time, memory, output = time_with_cuda_events(lambda: wrapper(inputs))
                prof.step()
            times.append(time)
            memories.append(memory)
    times = np.array(times)
    memories = np.array(memories)

    prof.export_chrome_trace('trace.json')
    return {
        'mean_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'std_time': np.std(times),
        'mean_memory': np.mean(memories),
        'min_memory': np.min(memories),
        'max_memory': np.max(memories),
        'std_memory': np.std(memories),
        'output': output
    }


def get_fps(t):
    return 1. / t

def print_results(res, header=None):
    if header:
        print(header)
    print(f"Mean: {res['mean_time']:.2f}ms, {res['mean_memory']:.2f}MB")
    print(f"Min: {res['min_time']:.2f}ms, {res['min_memory']:.2f}MB")
    print(f"Max: {res['max_time']:.2f}ms, {res['max_memory']:.2f}MB")
    print(f"Std: {res['std_time']:.2f}ms, {res['std_memory']:.2f}MB")
    print(f"Avg FPS: {1000./res['mean_time']:.2f}FPS")