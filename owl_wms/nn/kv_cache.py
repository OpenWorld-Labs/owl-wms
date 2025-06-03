import torch
from torch import nn

from ..configs import TransformerConfig

class KVCache:
    def __init__(self, config : TransformerConfig):
        self.shape = None
        self.config = config

        self.cache = None
        self.device = 'cuda'
        self.dtype = torch.bfloat16
        
        self.should_update = False

    def enable_cache_updates(self):
        self.should_update = True
    
    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

    def reset(self, batch_size = 1):
        self.shape = (batch_size, self.config.n_heads, 0, self.config.d_model//self.config.n_heads)
        dummy = torch.empty(*self.shape, device = self.device, dtype = self.dtype)
        self.cache = [(torch.empty_like(dummy), torch.empty_like(dummy)) for _ in range(self.config.n_layers)]

    @torch.no_grad()
    def get(self, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        return k,v
    
    @torch.no_grad()
    def push(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind] # each [b,h,n,d]
        k = torch.cat([k,new_k],dim=2)
        v = torch.cat([v,new_v],dim=2)
        self.cache[layer_ind] = (k,v)
    
    @torch.no_grad()
    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"
        self.cache[layer_ind] = (new_k,new_v)

    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0].shape[2]

    def shape(self):
        return self.shape