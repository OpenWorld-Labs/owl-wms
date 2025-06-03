import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm, RMSNorm, QKNorm
from .mlp import MLP

import einops as eo

from .modulation import AdaLN, Gate
#from .embeddings import FlatVideoRoPE
from rotary_embedding_torch import RotaryEmbedding

torch.backends.cuda.enable_flash_sdp(enabled = True)

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

def create_block_causal_mask(tokens, tokens_per_frame):
    frames = tokens // tokens_per_frame
    
    # Create base causal mask
    mask = torch.zeros(tokens, tokens)
    
    # Allow attention within each frame
    for i in range(frames):
        start = i * tokens_per_frame
        end = (i + 1) * tokens_per_frame
        mask[start:end, :end] = True
        
    return mask

class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.layer_ind = None

        self.tokens_per_frame = config.tokens_per_frame
        self.causal = config.causal

    def forward(self, x, kv_cache = None):
        q,k,v = eo.rearrange(self.qkv(x), 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)
        q,k = self.qk_norm(q,k)

        if not self.causal or len(kv_cache) > 0:
            mask = None
        else:
            mask = create_block_causal_mask(x.shape[1], self.tokens_per_frame).to(x.device)
            mask = mask.unsqueeze(0).repeat(x.shape[0], 1, 1)

        if kv_cache is not None:
            old_k, old_v = kv_cache.get(self.layer_ind)

            len_k = old_k.shape[2]

            new_k = torch.cat([old_k, k], dim = 2).contiguous()
            new_v = torch.cat([old_v, v], dim = 2).contiguous()

            if kv_cache.should_update:
                kv_cache.update(new_k, new_v, self.layer_ind)

            # Add rope here if we do use it
            x = F.scaled_dot_product_attention(q, new_k, new_v, attn_mask = mask)
            x = x[:,:,-q.shape[2]:] # Skip cached outputs (not relevant now)
        else:
            x = F.scaled_dot_product_attention(q,k,v, attn_mask = mask)

        x = eo.rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond, kv_cache = None):
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x, kv_cache)
        x = self.gate1(x, cond)
        x = res1 + x
        
        res2 = x.clone()
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = res2 + x

        return x

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, cond, kv_cache = None):
        for block in self.blocks:
            x = block(x, cond, kv_cache)

        return x

class UViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i

        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(nn.Linear(config.d_model * 2, config.d_model))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond, kv_cache = None):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]
            
            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)
            
            x = self.blocks[i](x, cond, kv_cache)

        return x

# === VIT Specific Layers ===

class FinalLayer(nn.Module):
    def __init__(self, sample_size, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels*patch_size*patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)

        return x

def test_attn_mask():
    total_tokens = 64 
    tokens_per_frame = 8

    # Block causal mask
    mask = create_block_causal_mask(total_tokens, tokens_per_frame)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    plt.imshow(mask.float().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.title(f'Block Causal Mask ({total_tokens} tokens, {tokens_per_frame} per frame)')
    plt.xlabel('Query Position')
    plt.ylabel('Key Position')
    plt.savefig('test_mask.png')
    plt.close()

    # Basic causal mask
    import torch
    basic_mask = torch.tril(torch.ones(total_tokens, total_tokens))
    
    plt.figure(figsize=(10,10))
    plt.imshow(basic_mask.numpy(), cmap='gray')
    plt.colorbar()
    plt.title('Basic Causal Mask (tril)')
    plt.xlabel('Query Position')
    plt.ylabel('Key Position')
    plt.savefig('test_basic_mask.png')
    plt.close()

@torch.no_grad()
def test_kv_cache():
    from .kv_cache import KVCache
    from ..configs import TransformerConfig

    # Create test configs
    config = TransformerConfig(
        n_layers=2,
        n_heads=8, 
        d_model=64,
        tokens_per_frame=8
    )

    # Create model and cache
    model = DiT(config).cuda()
    cache = KVCache(config)
    cache.to('cuda')

    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model).cuda()
    cond = torch.randn(batch_size, seq_len//config.tokens_per_frame, config.d_model).cuda()

    # Test forward pass with cache
    cache.reset(batch_size)
    cache.enable_cache_updates()

    # First forward pass should populate cache
    out1 = model(x, cond, cache)
    
    # Second pass should use cached values
    cache.disable_cache_updates()
    out2 = model(x, cond, cache)

    # Outputs should match
    print("Max difference between outputs:", torch.max(torch.abs(out1 - out2)).item())
    print("Cache test complete")

if __name__ == "__main__":
    test_kv_cache()
    