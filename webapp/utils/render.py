import torch
import math
import time
from datetime import datetime
from pathlib import Path
import imageio
import numpy as np

import einops as eo

from webapp.utils.models import load_models
from webapp.utils.samplers import create_sampler
from webapp.utils.action_builder import ActionSequenceGenerator, ActionConfig, ActionPattern

HEIGHT = 256
WIDTH = 256
D_MODEL = 1024
CHANNELS = 128
SEQUENCE_LENGTH = 60
TOKENS_PER_FRAME = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "generated_videos"
SAMPLER_TYPE = 'window'
DEFAULT_PATTERN = ActionPattern.LOOK_AROUND


def setup_output_dir(): Path(OUTPUT_DIR).mkdir(exist_ok=True)


def generate_dummy_actions(pattern=DEFAULT_PATTERN, length=SEQUENCE_LENGTH):
    """Generate dummy actions for video conditioning."""
    config = ActionConfig(
        sequence_length=length,
        device=DEVICE,
        dtype=torch.float32
    )
    
    generator = ActionSequenceGenerator(config)
    mouse, buttons = generator.generate_pattern(pattern)
    
    # Add batch dimension
    return mouse.unsqueeze(0), buttons.unsqueeze(0)


def synthesize_video(mouse_actions, button_actions, encoder, decoder, sampler):
    """Generate video using model and sampler."""
    batch_size, sequence_length = mouse_actions.shape[:2]

    dummy_batch = torch.randn(
        batch_size, sequence_length, CHANNELS,
        int(math.sqrt(TOKENS_PER_FRAME)), # H
        int(math.sqrt(TOKENS_PER_FRAME)), # W
        device=DEVICE, dtype=torch.float32
    )
    
    # Ensure actions are on correct device
    mouse_actions = mouse_actions.to(DEVICE)
    button_actions = button_actions.to(DEVICE)
    
    # Generate video
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latents, video = sampler(
            dummy_batch=dummy_batch,
            mouse=mouse_actions,
            btn=button_actions
        )
    
    return video


def save_video(video_tensor: torch.Tensor, filename="generated_video", fps=30):
    """
    Save video tensor as MP4 file.
    
    Args:
        video_tensor: Tensor with shape [batch_size, sequence_length, channels, height, width]
                     Expected range: [-1, 1] (VAE decoder output)
        filename: Base filename (without extension)
        fps: Frames per second for output video
        
    Returns:
        str: Path to saved video file
    """
    setup_output_dir()
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"{filename}_{timestamp}.mp4"
    
    # Convert tensor to numpy and handle batch dimension
    video_np: np.ndarray = video_tensor.float().cpu().detach().numpy()
    
    # Take first batch item if batch_size > 1
    if video_np.ndim == 5:  # [batch, seq, channels, height, width]
        video_np = video_np[0]  # Take first batch item: [seq, channels, height, width]
    
    # Convert from [seq, channels, height, width] to [seq, height, width, channels]
    video_np = eo.rearrange(video_np, 'seq c h w -> seq h w c')
    
    # Denormalize from actual range to [0,255]
    print(f'Video range before denorm: [{video_np.min():.3f}, {video_np.max():.3f}]')
    
    # Normalize to [0, 1] using actual min/max
    video_min, video_max = video_np.min(), video_np.max()
    if video_max > video_min:  # Avoid division by zero
        video_np = (video_np - video_min) / (video_max - video_min)  # -> [0, 1]
    else:
        video_np = np.zeros_like(video_np)  # Handle edge case of constant values
    
    video_np = (video_np * 255.0).clip(0, 255).astype(np.uint8)  # [0,1] -> [0,255]
    
    # Handle grayscale (single channel) by converting to RGB
    if video_np.shape[-1] == 1:
        video_np = np.repeat(video_np, 3, axis=-1)
    elif video_np.shape[-1] > 3:
        print(f'Warning: video has {video_np.shape[-1]} channels, taking only first 3')
        video_np = video_np[:, :, :, :3]
    
    try:
        imageio.mimsave(output_path, video_np, fps=fps, codec='libx264')
        return str(output_path)
    except Exception as e:
        print(f"Warning: Could not save as MP4 ({e}), falling back to .pt format")

    # Fallback: save as PyTorch tensor
    fallback_path = Path(OUTPUT_DIR) / f"{filename}_{timestamp}.pt"
    torch.save(video_tensor.cpu(), fallback_path)
    return str(fallback_path)


def render_video(pattern=DEFAULT_PATTERN, length=SEQUENCE_LENGTH, verbose=True):
    """Simple video rendering - just load, generate, save."""
    if verbose:
        print(f"🎬 Rendering video with pattern: {pattern.value}")
    
    # Load model and create sampler
    if verbose:
        print("Loading model...")
    encoder, decoder, model_config = load_models(device=DEVICE, verbose=verbose)
    
    if verbose:
        print("Creating sampler...")
    sampler = create_sampler(SAMPLER_TYPE, encoder, decoder)
    
    # Generate actions
    if verbose:
        print("Generating actions...")
    mouse, buttons = generate_dummy_actions(pattern, length)
    
    # Synthesize video
    if verbose:
        print("Synthesizing video...")
    video = synthesize_video(mouse, buttons, encoder, decoder, sampler)
    
    # Save video
    if verbose:
        print("Saving video...")
    path = save_video(video, f"render_{pattern.value}")
    
    if verbose:
        print(f"✅ Done! Video saved to: {path}")
        print(f"   Video shape: {video.shape}")
    
    return path


if __name__ == "__main__":
    # Simple usage examples
    print("🎮 Simple OWL-WMS Video Renderer")
    
    # Render with default settings
    render_video(verbose=True)
    
    # Render with different pattern
    render_video(ActionPattern.AIM_AND_SHOOT, verbose=True)
    render_video(ActionPattern.LOOK_AROUND, verbose=True)
    render_video(ActionPattern.CIRCLE_STRAFE, verbose=True)
