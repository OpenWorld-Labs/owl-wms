import math
import time
import torch
import asyncio
from torch          import nn

from webapp.utils.samplers                import create_sampler
from webapp.utils.configs           import SamplingConfig, StreamingConfig
from owl_wms.configs                import TrainingConfig, TransformerConfig as ModelConfig

class FrameBuffer:
    """Manages frame streaming at precise timing."""
    
    def __init__(self, streaming_config: StreamingConfig):
        self.streaming_config = streaming_config
        self.frame_queue = asyncio.Queue(maxsize=streaming_config.frames_per_batch * 2)  # Buffer 2 batches
        self.last_frame_time = 0.0
        
    async def add_frame_batch(self, frame_batch: torch.Tensor):
        """Add a batch of frames to the streaming queue."""
        # frame_batch shape: [1, frames_per_batch, channels, height, width]
        batch_size, num_frames = frame_batch.shape[:2]
        
        for i in range(num_frames):
            frame = frame_batch[0, i]  # take first batch cause 1 user only, then [channels, height, width]
            await self.frame_queue.put(frame)
    
    async def get_next_frame(self) -> torch.Tensor:
        """Get next frame for streaming at precise timing."""
        # Calculate when to release the next frame
        now = time.time()
        time_since_last = now - self.last_frame_time
        time_to_wait = max(0, self.streaming_config.frame_interval - time_since_last)
        
        if time_to_wait > 0:
            await asyncio.sleep(time_to_wait)
        
        frame = await self.frame_queue.get()
        self.last_frame_time = time.time()
        return frame

class StreamingFrameGenerator:
    """Wraps WindowCFGSampler for real-time 8-frame batch generation."""
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 streaming_config: StreamingConfig,
                 model_config: ModelConfig,
                 train_config: TrainingConfig,
                 sampling_config: SamplingConfig):
        self.streaming_config = streaming_config
        self.model_config     = model_config
        self.train_config     = train_config
        self.sampling_config  = sampling_config

        self.encoder = encoder
        self.decoder = decoder
        
        # Create WindowCFGSampler for 8-frame generation
        self.sample_window_fn = create_sampler('window', encoder, decoder,
                                             batch_size=1,
                                             sampling_steps=self.sampling_config.sampling_steps,
                                             cfg_scale=self.sampling_config.cfg_scale,
                                             scale=self.train_config.vae_scale)
        # Initialize frame history as empty tensor
        self.frame_history: torch.Tensor = torch.tensor([], device=self.streaming_config.device)

    def add_to_history(self, frame_batch: torch.Tensor):
        if self.frame_history.equal(torch.tensor([], device=self.streaming_config.device)):
            self.frame_history = frame_batch
            return

        self.frame_history = torch.cat([self.frame_history, frame_batch], dim=1)
        
        # cap this at around 60 frames
        if self.frame_history.shape[1] > self.streaming_config.window_length:
            self.frame_history = self.frame_history[:, -self.streaming_config.window_length:]

    
    @property
    def dummy_batch(self) -> torch.Tensor:
        """Generate dummy autoencoder latents for cold start."""
        tokens_h = tokens_w = int(math.sqrt(self.model_config.tokens_per_frame))
        dummy_frames = torch.randn(
            1, self.streaming_config.window_length,
            self.model_config.channels, tokens_h, tokens_w,
            device=self.streaming_config.device, dtype=torch.bfloat16)
        return dummy_frames

    
    async def generate_frame_batch(self, mouse_batch: torch.Tensor, button_batch: torch.Tensor) -> torch.Tensor:
        """
        Generate window_length frames, return first frames_per_batch for streaming.
        
        Args:
            mouse_batch: [1, window_length, 2] 
            button_batch: [1, window_length, n_buttons]
            
        Returns:
            frame_batch: [1, frames_per_batch, 3, 256, 256] - only streaming frames
        """
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latents, full_frames = self.sample_window_fn(dummy_batch=self.dummy_batch, mouse=mouse_batch, btn=button_batch)  # [1, window_length, 3, 256, 256]
            self.add_to_history(latents[0, :self.streaming_config.frames_per_batch, :, :, :])

        # Take only first frames_per_batch for streaming
        streaming_frames = full_frames[:, :self.streaming_config.frames_per_batch, :, :, :]
        return streaming_frames  # [1, frames_per_batch, 3, 256, 256]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.encoder, self.decoder ; torch.cuda.empty_cache()
