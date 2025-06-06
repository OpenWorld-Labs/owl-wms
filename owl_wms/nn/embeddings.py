import torch
from torch import nn
import torch.nn.functional as F

import einops as eo
from .mlp import MLPCustom

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import einops as eo

class VideoRoPE(nn.Module):
    """
    Video RoPE embedding for when latents are 3D [n,h,w]
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//8,
            freqs_for = 'pixel',
            max_freq = 256
        )
        n_patches = config.sample_size // config.patch_size
        self.tokens_per_frame = n_patches**2

        self.rearrange_in = lambda x: eo.rearrange(x, 'b h (n_t n_y n_x) d -> b h n_t n_y n_x d', n_y = n_patches)
        self.rearrange_out = lambda x: eo.rearrange(x, 'b h n_t n_y n_x d -> b h (n_t n_y n_x) d')
        self.get_freqs = lambda n_t: self.pos_emb.get_axial_freqs(n_t, n_patches, n_patches)

    def forward(self, q, k):
        # q k both [b,h,n,d]
        q = self.rearrange_in(q)
        k = self.rearrange_in(k)

        n_t = q.shape[2]
        freqs = self.get_freqs(n_t)

        q = apply_rotary_emb(freqs.float(), q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs.float(), k.float()).to(k.dtype)

        q = self.rearrange_out(q)
        k = self.rearrange_out(k)
        
        return q, k

class FlatVideoRoPE(nn.Module):
    """
    Video RoPE embedding for when latents are 2d [n,m] (1D Frame Tokenization)
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//4,
            freqs_for = 'pixel',
            max_freq = 256
        )
        self.pos_emb.freqs.requires_grad = False
        self.tokens_per_frame = config.sample_size

        self.rearrange_in = lambda x: eo.rearrange(x, 'b h (n_t m) d -> b h n_t m d', m = self.tokens_per_frame)
        self.rearrange_out = lambda x: eo.rearrange(x, 'b h n_t m d -> b h (n_t m) d')
        self.get_freqs = lambda n_t: self.pos_emb.get_axial_freqs(n_t, self.tokens_per_frame)

    def forward(self, q, k):
        # q k both [b,h,n,d]
        q = self.rearrange_in(q)
        k = self.rearrange_in(k)

        n_t = q.shape[2]
        with torch.no_grad():
            freqs = self.get_freqs(n_t)

        q = apply_rotary_emb(freqs.float(), q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs.float(), k.float()).to(k.dtype)

        q = self.rearrange_out(q)
        k = self.rearrange_out(k)
        
        return q, k


class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()
        # n_seq is tokens_per_frame * n_frames, dim is d_model
        self.p = nn.Parameter(torch.randn(n_seq,dim)*0.02)

    def forward(self, x):
        b,n,d = x.shape
        p = eo.repeat(self.p, 'n d -> b n d', b = b)
        return x + p

class SinCosEmbed(nn.Module):
    def __init__(self, dim, theta=300, mult=1000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.mult = mult

    def forward(self, x):
        # Handle different input types
        if isinstance(x, float):
            x = torch.tensor([x])
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
            
        # Handle [b,n] inputs
        reshape_out = False
        if x.dim() == 2:
            b, n = x.shape
            x = x.reshape(b*n)
            reshape_out = True
            
        x = x * self.mult
        
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.theta)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        
        # Match device and dtype of input
        emb = emb.to(device=x.device, dtype=x.dtype)
        
        # Compute sin/cos embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        
        # Reshape back if needed
        if reshape_out:
            emb = emb.reshape(b, n, -1)
            
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sincos = SinCosEmbed(512, theta=300, mult = 1000)
        self.mlp = MLPCustom(512, dim * 4, dim)
    
    def forward(self, x):
        x = self.sincos(x)
        x = self.mlp(x)
        return x

class ConditionEmbedding(nn.Module):
    def __init__(self, n_classes, dim):
        super().__init__()
        
        self.embedding = nn.Embedding(n_classes, dim)
        self.mlp = MLPCustom(dim, dim * 4, dim)
    
    def forward(self, x):
        # x is long tensor of [b,]
        x = self.embedding(x)
        x = self.mlp(x)
        return x

class MouseEmbedding(nn.Module):
    def __init__(self, dim_out, dim=512):
        super().__init__()
        
        # For angle embeddings
        self.angle_proj = nn.Linear(2, dim//2, bias=False)
        
        # For magnitude embeddings
        self.magnitude_embed = SinCosEmbed(dim//2)
        
        # Final MLP
        self.mlp = MLPCustom(dim, dim * 4, dim_out)

    def forward(self, x):
        # x is [b,n,2]
        # Convert to polar coordinates
        with torch.no_grad():
            # Apply symlog scaling to x and y coordinates
            x_sign = torch.sign(x)
            x_abs = torch.abs(x)
            x = x_sign * torch.log1p(x_abs)

            angles = torch.atan2(x[..., 1], x[..., 0])  # [b,n]
            magnitudes = torch.norm(x, dim=-1)  # [b,n]
            
            # Embed angles and magnitudes
            angle_emb = torch.stack([
                torch.cos(angles),
                torch.sin(angles)
            ], dim=-1)  # [b,n,2]
            magnitude_emb = self.magnitude_embed(magnitudes)  # [b,n,dim//2]

        angle_emb = self.angle_proj(angle_emb)  # [b,n,dim//2]
        
        # Combine and pass through MLP
        x = torch.cat([angle_emb, magnitude_emb], dim=-1)  # [b,n,dim]
        x = self.mlp(x)
        return x

class ButtonEmbeddding(nn.Module):
    def __init__(self, n_buttons, dim_out, dim=512):
        super().__init__()

        self.proj = MLPCustom(n_buttons, dim*4, dim_out)
    
    def forward(self, x):
        # x is float tensor of 0s and 1s
        x = (x * 2) - 1
        x = self.proj(x)
        return x

class ControlEmbedding(nn.Module):
    def __init__(self, n_buttons, dim_out, dim = 512):
        super().__init__()

        self.mouse = MouseEmbedding(dim_out, dim)
        self.button = ButtonEmbeddding(n_buttons, dim_out, dim)

    def forward(self, mouse, button):
        # mouse : [b,n,2]
        # button : [b,n,n_buttons]

        # out is [b,n,d]

        return self.mouse(mouse) + self.button(button)
