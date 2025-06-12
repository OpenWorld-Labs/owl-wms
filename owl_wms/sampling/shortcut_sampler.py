import cv2
import math
import pathlib
import torch
from torch import Module
from tqdm import tqdm
from typing import Optional

from ..nn.kv_cache import KVCache
from ..utils import batch_permute_to_length
from ..models.gamerft_shortcut import ShortcutGameRFT


def zlerp(x, alpha):
    z = torch.randn_like(x)
    return x * (1. - alpha) + z * alpha

def load_mp4_as_tensor(mp4_path: pathlib.Path) -> torch.Tensor:
    """Load MP4 as tensor in format [N, C=3, H, W] with values in [-1, 1]"""
    video = cv2.VideoCapture(str(mp4_path))
    
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {mp4_path}")
    
    frames = []
    while True:
        ret, frame = video.read()
        if not ret: 
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to torch tensor and normalize to [-1, 1]
        frame = torch.from_numpy(frame).float() / 127.5 - 1.0
        
        # Rearrange from [H, W, C] to [C, H, W]
        frame = frame.permute(2, 0, 1)
        
        frames.append(frame)
    
    video.release()
    
    if not frames:
        raise ValueError(f"No frames found in video: {mp4_path}")
    
    # Stack to [N, C, H, W]
    return torch.stack(frames)


class InferenceCachedShortcutSampler:
    
    ALPHA = 0.25

    def __init__(self,
                 model: ShortcutGameRFT,
                 window_length  = 60,
                 num_frames     =  1,
                 only_return_generated = False,
                 vae_scale = 2.17,
                 decode_fn: Optional[Module] = None,
                 initial_history_pt_path: Optional[pathlib.Path] = None,
                 initial_history_mp4_path: Optional[pathlib.Path] = None,
                 encoder: Optional[Module] = None):
        # -- 
        self.model: ShortcutGameRFT = model
        self.window_length          = window_length
        self.num_frames             = num_frames
        
        self.vae_scale              = vae_scale
        self.only_return_generated  = only_return_generated

        # -- 
        self._cache_built = False
        self.cache = KVCache(model.config)
        self.decode_fn = decode_fn
        self.initial_history_pt_path = initial_history_pt_path
        self.initial_history_mp4_path = initial_history_mp4_path
        self.encoder = encoder
        
        assert initial_history_pt_path is not None or initial_history_mp4_path is not None, \
            'Either initial_history_pt_path or initial_history_mp4_path must be provided'
        
        if initial_history_mp4_path is not None:
            assert encoder is not None, \
                'Encoder must be provided if initial_history_mp4_path is provided'

        self.initial_history_bWchw = self.init_history(self.initial_history_pt_path, self.initial_history_mp4_path)
        self.keyframe_b1chw        = self.initial_history_bWchw[:,0]

    def init_history(self,
                     initial_history_pt_path: pathlib.Path | None,
                     initial_history_mp4_path: pathlib.Path | None) -> torch.Tensor:

        if initial_history_pt_path is not None:
            history_wchw = torch.load(initial_history_pt_path)
        else:
            history_wrgb = load_mp4_as_tensor(initial_history_mp4_path).unsqueeze(0) # add batch dim
            history_wchw = self.encoder(history_wrgb)
            # NOTE This is so we avoid generating the history with a compiled model.
            torch.save(history_wchw, initial_history_mp4_path.absolute().replace('.mp4', '.pt'))

        N = self.window_length
        C = self.model.config.channels
        H = W = int(math.sqrt(self.model.config.tokens_per_frame))
        
        assert tuple(history_wchw.shape) == (1, N, C, H, W), \
            f'Initial history must have shape (B=1, {N=}, {C=}, {H=}, {W=}), ' \
            f'but got {tuple(history_wchw.shape)}'
        
        return history_wchw


    def init_cache(self,
                   frames_bWchw,   # [B, W, c, h, w] - NOTE history of frames
                   keyframe_b1chw, # [B, 1, c, h, w] - NOTE keyframe conditioning
                   mouse_bW2,      # [B, W, 2]
                   button_bW11,    # [B, W, 11]
                   ts_bW,          # [B, W]
                   d_bW):          # [B, W]
        if self._cache_built:
            print(f'WARNING: Cache already built but called `init_cache` again - ignoring.')
            return
        
        B, N, *_ = frames_bWchw.shape
        
        self.cache.reset(B) ; self.cache.enable_cache_updates()

        # -- noise the history and fwd to kv cache
        self.model.core.sample(x=zlerp(frames_bWchw, self.ALPHA),
                               y=keyframe_b1chw,
                               mouse=mouse_bW2,
                               btn=button_bW11,
                               cache=self.cache,
                               ts=ts_bW, d=d_bW)

        self.cache.disable_cache_updates() ; self._cache_built = True
        print(f'Cache initialized for {B} x {N} frames - {[[i.shape for i in elt]
                                                            for elt in self.cache.cache]}')
        return self.cache

    def __call__(self,
            ctxt_frame_b1chw, # [B, 1, c, h, w] - NOTE Keyframe conditioning
            mouse_b1_2,      # [B, 1, 2] - NOTE mouse actions
            button_b1_11,    # [B, 1, 11] - NOTE button actions
            ts_alpha_b1,     # [B, 1] - NOTE overall denoising timestamp (e.g. 128)
            d_alpha_b1,      # [B, 1] - NOTE denoising step budget (e.g. 4)
        ) -> torch.Tensor:  # [B, 1, c, h, w]
        # 1. ---- generate next frame ----
        self.cache.disable_cache_updates()
        # 1.A) -- use the full context, including entire action history, to generate the next frame given cache. 
        frame           = self.model.core.sample(None, ctxt_frame_b1chw,
                                                mouse_b1_2, button_b1_11,
                                                self.cache, ts=None, d=None)  # NOTE simulating one-step sampling
        # 2. ---- repopulate cache ----
        self.cache.enable_cache_updates() ; self.cache.truncate(1)
        self.model.core.sample( x=zlerp(frame, self.ALPHA),  # diffuse with noised frame to repopulate cache
                                y=ctxt_frame_b1chw,
                                mouse=mouse_b1_2,
                                btn=button_b1_11,
                                cache=self.cache,
                                ts=ts_alpha_b1, d=d_alpha_b1)
        self.cache.disable_cache_updates()
        return frame

    @torch.no_grad()
    def generate_frames(self,
            history_bWchw,  # [B, W, c, h, w] - NOTE: MP4 from CoD initially, and after that it's just KV cache. 
            mouse_bT2,      # [B, W+N, 2] - Actions taken by the user.
            button_bT11,    # [B, W+N, 11] - Actions taken by the user.
        ) -> torch.Tensor:  # [B, W+N, c, h, w] - either latent or rgb.

        if not self._cache_built:
            print(f'WARNING: Cache not built, but called `generate_frames` - initializing cache.')
            self.init_cache(history_bWchw, self.keyframe_b1chw, mouse_bT2, button_bT11)

        # If does not have batch-size, add it. This sampler is going to be used for single-user inference so batch-size is always 1.
        # The caller might not specify the batch-size, so we have this here.
        if history_bWchw.ndim == 4:
            history_bWchw = history_bWchw.unsqueeze(1)

        history_bWchw = history_bWchw[:, -self.window_length:, ::]

        assert history_bWchw.shape[1] == self.window_length, \
            f'Window history must be at least {self.window_length} frames long, but got {history_bWchw.shape}'

        ts_alpha_bW = torch.ones_like(history_bWchw[:,:,0,0,0]) * self.ALPHA
        d_alpha_bW  = torch.ones_like(history_bWchw[:,:,0,0,0]) * round(1./self.ALPHA)

        ts_alpha_b1 = ts_alpha_bW[:,0].unsqueeze(1)
        d_alpha_b1  = d_alpha_bW [:,0].unsqueeze(1)

        frames_latent = []
        for frame_idx in range(self.num_frames):
            btn_atom        = button_bT11[:, self.window_length+frame_idx].unsqueeze(1)
            mouse_atom      = mouse_bT2  [:, self.window_length+frame_idx].unsqueeze(1)
            frame           = self.__call__(ctxt_frame_b1chw=self.keyframe_b1chw,
                                           mouse_b1_2=mouse_atom, button_b1_11=btn_atom,
                                           ts_alpha_b1=ts_alpha_b1, d_alpha_b1=d_alpha_b1)
            frames_latent += [frame]

        frames_latent = torch.cat(frames_latent, dim=1)

        if self.only_return_generated: frames_latent = frames_latent[:,-self.num_frames:]

        if self.decode_fn is not None:
            frames_rgb = self.decode_fn(frames_latent * self.vae_scale)
            return frames_rgb, mouse_bT2, button_bT11

        return frames_latent, mouse_bT2, button_bT11


class CacheShortcutSampler:
    """
    Shortcut CFG sampler builds cache with 4 step diffusion.
    Samples new frames in 1 step.

    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, window_length = 60, num_frames = 60, only_return_generated = False):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.window_length = window_length
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, history, keyframe, mouse, btn, decode_fn = None, scale = 1):
        # dummy_batch is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        history = history[:,:self.window_length]
        new_frames = []
        alpha = 0.25 # This number is special for our sampler

        # Extended fake controls to use during sampling
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, num_frames + self.window_length)

        # Generate cache over history
        noisy_history = zlerp(history.clone(), alpha)
        ts = torch.ones_like(noisy_history[:,:,0,0,0]) * alpha
        d = torch.ones_like(noisy_history[:,:,0,0,0]) * round(1./alpha)
        ts_single = ts[:,0].unsqueeze(1)
        d_single = d[:,0].unsqueeze(1)

        cache = KVCache(model.config)
        cache.reset(history.shape[0])

        cache.enable_cache_updates()
        _ = model.sample(noisy_history, keyframe, mouse, btn, cache, ts, d)
        cache.disable_cache_updates()

        # Cache is now built!
        
        for frame_idx in tqdm(range(num_frames)):
            cache.truncate(1) # Drop first frame

            # Generate new frame
            cache.disable_cache_updates()
            mouse = extended_mouse[:,self.window_length+frame_idx].unsqueeze(1)
            btn = extended_btn[:,self.window_length+frame_idx].unsqueeze(1)
            # N+1
            new_frame = model.sample(None, keyframe, mouse, btn, cache) # [b,1,c,h,w]
            new_frames.append(new_frame)
            
            # Add that frame to the cache
            cache.enable_cache_updates()
            new_frame_noisy = zlerp(new_frame, alpha)
            # N+2, noisy(N+1) gets cached
            _ = model.sample(new_frame_noisy, keyframe, mouse, btn, cache, ts_single, d_single)

        new_frames = torch.cat(new_frames, dim = 1)
        x = torch.cat([history,new_frames], dim = 1)

        if self.only_return_generated:
            x = x[:,-num_frames:]
            extended_mouse = extended_mouse[:,-num_frames:]
            extended_btn = extended_btn[:,-num_frames:]

        if decode_fn is not None:
            x = x * scale 
            x = decode_fn(x)
    
        return x, extended_mouse, extended_btn

class WindowShortcutSampler:
    """
    Same as above but with no cache

    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, window_length = 60, num_frames = 60, only_return_generated = False):
        self.window_length = window_length
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, history, keyframe, mouse, btn, decode_fn = None, scale = 1):
        # history is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        history = history[:,:self.window_length]
        new_frames = []
        alpha = 0.25 # This number is special for our sampler

        # Extended fake controls to use during sampling
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, self.num_frames + self.window_length)

        # Initialize window history
        window_history = history.clone()

        for frame_idx in tqdm(range(self.num_frames)):
            # Setup window history
            x = window_history[:,-self.window_length:].clone()
            
            # Noise all but last frame to alpha
            x[:,:-1] = zlerp(x[:,:-1], alpha)
            # Last frame starts as random noise
            x[:,-1] = torch.randn_like(x[:,-1])

            # Setup timesteps - alpha for context, 1.0 for generated
            ts = torch.ones_like(x[:,:,0,0,0])
            ts[:,:-1] = alpha
            
            # Setup diffusion steps - 4 for context, 1 for generated
            d = torch.ones_like(x[:,:,0,0,0])
            d[:,:-1] = 4

            # Get current controls
            curr_mouse = extended_mouse[:,frame_idx:frame_idx+self.window_length]
            curr_btn = extended_btn[:,frame_idx:frame_idx+self.window_length]

            # Generate new frame
            pred = model.sample(x, keyframe, curr_mouse, curr_btn, None, ts, d)
            new_frame = pred[:,-1:] # Take only the last frame
            new_frames.append(new_frame)
            
            # Add new frame to window history
            window_history = torch.cat([window_history, new_frame], dim=1)

        new_frames = torch.cat(new_frames, dim=1)
        x = torch.cat([history, new_frames], dim=1)

        if self.only_return_generated:
            x = x[:,-self.num_frames:]
            extended_mouse = extended_mouse[:,-self.num_frames:]
            extended_btn = extended_btn[:,-self.num_frames:]

        if decode_fn is not None:
            x = x * scale
            x = decode_fn(x)

        return x, extended_mouse, extended_btn

<<<<<<< HEAD

class InferenceWindowShortcutSamplerNoKeyframe:
    """
    Window-based shortcut sampler without keyframe conditioning or KV cache.
    Generates frames using sliding window approach with diffusion forcing.
    
    :param model: The shortcut diffusion model
    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample per generate_frames call
    :param only_return_generated: Whether to only return the generated frames
    :param vae_scale: Scale factor for VAE decoding
    :param decode_fn: Optional decoder function
    :param initial_history_pt_path: Path to pre-encoded initial history tensor
    :param initial_history_mp4_path: Path to MP4 file for initial history
    :param encoder: Encoder module (required if using MP4 path)
    """
    
    ALPHA = 0.25  # Noise level for context frames (step 3 of 4-step diffusion)

    def __init__(self,
                 model,
                 window_length = 60,
                 num_frames = 1,
                 only_return_generated = False,
                 vae_scale = 2.17,  # TODO Shab trained a new VAE so this needs to be updated.
                 decode_fn: Optional[Module] = None,
                 initial_history_pt_path: Optional[pathlib.Path] = None,
                 initial_history_mp4_path: Optional[pathlib.Path] = None,
                 encoder: Optional[Module] = None):
        
        self.model = model
        self.window_length = window_length
        self.num_frames = num_frames
        self.vae_scale = vae_scale
        self.only_return_generated = only_return_generated
        self.decode_fn = decode_fn
        
        self.initial_history_pt_path = initial_history_pt_path
        self.initial_history_mp4_path = initial_history_mp4_path
        self.encoder = encoder
        
        assert initial_history_pt_path is not None or initial_history_mp4_path is not None, \
            'Either initial_history_pt_path or initial_history_mp4_path must be provided'
        
        if initial_history_mp4_path is not None:
            assert encoder is not None, \
                'Encoder must be provided if initial_history_mp4_path is provided'

        self.initial_history_bWchw = self.init_history(self.initial_history_pt_path, self.initial_history_mp4_path)

    def init_history(self,
                     initial_history_pt_path: pathlib.Path | None,
                     initial_history_mp4_path: pathlib.Path | None) -> torch.Tensor:
        """Initialize history from either .pt file or MP4 file"""
        
        if initial_history_pt_path is not None:
            history_wchw = torch.load(initial_history_pt_path)
        else:
            history_wrgb = load_mp4_as_tensor(initial_history_mp4_path).unsqueeze(0)  # add batch dim
            history_wchw = self.encoder(history_wrgb)
            # Save encoded version to avoid re-encoding
            torch.save(history_wchw, initial_history_mp4_path.absolute().replace('.mp4', '.pt'))

        N = self.window_length
        C = self.model.config.channels if hasattr(self.model, 'config') else history_wchw.shape[2]
        H = W = int(math.sqrt(self.model.config.tokens_per_frame)) if hasattr(self.model, 'config') else history_wchw.shape[3]
        
        assert tuple(history_wchw.shape) == (1, N, C, H, W), \
            f'Initial history must have shape (B=1, {N=}, {C=}, {H=}, {W=}), ' \
            f'but got {tuple(history_wchw.shape)}'
        
        return history_wchw

    def __call__(self,
                 window_history_bWchw,  # [B, W, c, h, w] - Current window of frames
                 mouse_bW2,             # [B, W, 2] - Mouse actions for window
                 button_bW11,           # [B, W, 11] - Button actions for window
                 ) -> torch.Tensor:     # [B, 1, c, h, w] - Generated frame
        """Generate a single frame given current window history"""
        
        # Setup window for generation
        x = window_history_bWchw[:, -self.window_length:].clone()
        
        # Noise all but last frame to alpha level (diffusion forcing)
        x[:, :-1] = zlerp(x[:, :-1], self.ALPHA)
        # Last frame starts as random noise
        x[:, -1] = torch.randn_like(x[:, -1])

        # Setup timesteps - ALPHA for context frames, 1.0 for generated frame
        ts = torch.ones_like(x[:, :, 0, 0, 0])
        ts[:, :-1] = self.ALPHA
        
        # Setup diffusion steps - 4 for context frames, 1 for generated frame
        d = torch.ones_like(x[:, :, 0, 0, 0])
        d[:, :-1] = 4  # Context frames use 4-step budget
        
        # Generate new frame using window
        pred = self.model.sample(x, mouse_bW2, button_bW11, None, ts, d)
        new_frame = pred[:, -1:]  # Take only the last (generated) frame
        
        return new_frame

    @torch.no_grad()
    def generate_frames(self,
                        history_bWchw,  # [B, W, c, h, w] - Initial history
                        mouse_bT2,      # [B, W+N, 2] - Mouse actions for entire sequence
                        button_bT11,    # [B, W+N, 11] - Button actions for entire sequence
                        ) -> torch.Tensor:  # [B, W+N, c, h, w] - Generated sequence
        """Generate multiple frames using sliding window approach"""
        
        # Handle batch dimension
        if history_bWchw.ndim == 4:
            history_bWchw = history_bWchw.unsqueeze(0)

        history_bWchw = history_bWchw[:, -self.window_length:]
        
        assert history_bWchw.shape[1] == self.window_length, \
            f'History must be exactly {self.window_length} frames long, but got {history_bWchw.shape[1]}'

        # Extended controls for generation
        extended_mouse, extended_btn = batch_permute_to_length(
            mouse_bT2[:, :self.window_length], 
            button_bT11[:, :self.window_length], 
            self.num_frames + self.window_length
        )

        # Initialize window history
        window_history = history_bWchw.clone()
        frames_latent = []

        for frame_idx in range(self.num_frames):
            # Get current window controls
            curr_mouse = extended_mouse[:, frame_idx:frame_idx + self.window_length]
            curr_btn = extended_btn[:, frame_idx:frame_idx + self.window_length]
            
            # Generate single frame
            new_frame = self.__call__(
                window_history_bWchw=window_history,
                mouse_bW2=curr_mouse,
                button_bW11=curr_btn
            )
            
            frames_latent.append(new_frame)
            
            # Add new frame to window history for next iteration
            window_history = torch.cat([window_history, new_frame], dim=1)

        # Combine all generated frames
        frames_latent = torch.cat(frames_latent, dim=1)
        
        # Combine with original history
        full_sequence = torch.cat([history_bWchw, frames_latent], dim=1)

        if self.only_return_generated:
            full_sequence = full_sequence[:, -self.num_frames:]
            extended_mouse = extended_mouse[:, -self.num_frames:]
            extended_btn = extended_btn[:, -self.num_frames:]

        if self.decode_fn is not None:
            frames_rgb = self.decode_fn(full_sequence * self.vae_scale)
            return frames_rgb, extended_mouse, extended_btn

        return full_sequence, extended_mouse, extended_btn
=======
class WindowShortcutSamplerNoKeyframe:
    """
    Same as above but with no cache

    :param window_length: Number of frames to use for each frame generation step
    :param num_frames: Number of new frames to sample
    :param only_return_generated: Whether to only return the generated frames
    """
    def __init__(self, window_length = 60, num_frames = 60, only_return_generated = False):
        self.window_length = window_length
        self.num_frames = num_frames
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, history, mouse, btn, decode_fn = None, scale = 1):
        # history is [b,n,c,h,w]
        # mouse is [b,n,2]
        # btn is [b,n,n_button]

        # output will be [b,n+self.num_frames,c,h,w]
        history = history[:,:self.window_length]
        new_frames = []
        alpha = 0.25 # This number is special for our sampler

        # Extended fake controls to use during sampling
        extended_mouse, extended_btn = batch_permute_to_length(mouse, btn, self.num_frames + self.window_length)

        # Initialize window history
        window_history = history.clone()

        for frame_idx in tqdm(range(self.num_frames)):
            # Setup window history
            x = window_history[:,-self.window_length:].clone()
            
            # Noise all but last frame to alpha
            x[:,:-1] = zlerp(x[:,:-1], alpha)
            # Last frame starts as random noise
            x[:,-1] = torch.randn_like(x[:,-1])

            # Setup timesteps - alpha for context, 1.0 for generated
            ts = torch.ones_like(x[:,:,0,0,0])
            ts[:,:-1] = alpha
            
            # Setup diffusion steps - 4 for context, 1 for generated
            d = torch.ones_like(x[:,:,0,0,0])
            d[:,:-1] = 4

            # Get current controls
            curr_mouse = extended_mouse[:,frame_idx:frame_idx+self.window_length]
            curr_btn = extended_btn[:,frame_idx:frame_idx+self.window_length]

            # Generate new frame
            pred = model.sample(x, curr_mouse, curr_btn, None, ts, d)
            new_frame = pred[:,-1:] # Take only the last frame
            new_frames.append(new_frame)
            
            # Add new frame to window history
            window_history = torch.cat([window_history, new_frame], dim=1)

        new_frames = torch.cat(new_frames, dim=1)
        x = torch.cat([history, new_frames], dim=1)

        if self.only_return_generated:
            x = x[:,-self.num_frames:]
            extended_mouse = extended_mouse[:,-self.num_frames:]
            extended_btn = extended_btn[:,-self.num_frames:]

        if decode_fn is not None:
            x = x * scale
            x = decode_fn(x)

        return x, extended_mouse, extended_btn
>>>>>>> causvid
