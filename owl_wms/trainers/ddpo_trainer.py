import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from collections import defaultdict
import time
from concurrent import futures
import numpy as np
import wandb
from functools import partial
import tqdm
from ema_pytorch import EMA

from .base import BaseTrainer
from ..utils import freeze, Timer, find_unused_params
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from ..muon import init_muon

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


class DDPOTrainer(BaseTrainer):
    """
    Trainer for DDPO (Denoising Diffusion Policy Optimization) for video world models.
    
    Implements reinforcement learning with human feedback (RLHF) for world models
    using PPO-style policy gradient optimization. World models take video frames,
    mouse movements, and button presses as inputs.
    
    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging  
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices
    :param local_rank: Rank for current device on this process
    :param world_size: Overall number of devices
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize model
        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)
        
        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")
        
        # Initialize training components
        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.total_step_counter = 0
        
        # Initialize video decoder
        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
        freeze(self.decoder)
        
        # DDPO specific components
        self.reward_fn = None  # Will be set by user
        
        # Initialize executor for async reward computation
        self.executor = futures.ThreadPoolExecutor(max_workers=2)
        
        # DDPO hyperparameters from train_cfg
        self.sampling_steps = self.train_cfg.get('sampling_steps', 64)
        self.num_train_timesteps = int(
            self.sampling_steps * self.train_cfg.get('timestep_fraction', 0.5)
        )
        self.clip_range = self.train_cfg.get('clip_range', 0.2)
        self.adv_clip_max = self.train_cfg.get('adv_clip_max', 5.0)
        
        # Sampling configuration
        self.sample_batch_size = self.train_cfg.get('sample_batch_size', 4)
        self.num_batches_per_epoch = self.train_cfg.get('num_batches_per_epoch', 4)
        self.num_inner_epochs = self.train_cfg.get('num_inner_epochs', 1)
        
    def set_reward_function(self, reward_fn):
        """Set the reward function for DDPO training."""
        self.reward_fn = reward_fn
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        torch.cuda.set_device(self.local_rank)
        
        # Prepare model
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Setup decoder
        self.decoder = self.decoder.cuda().eval().bfloat16()
        self.decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)
        
        # Setup EMA
        self.ema = EMA(
            self.model,
            beta=0.999,
            update_after_step=0,
            update_every=1
        )
        
        # Setup optimizer
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)
        
        # Setup scheduler
        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)
        
        # Setup gradient accumulation and scaler
        self.accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        self.accum_steps = max(1, self.accum_steps)
        self.scaler = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast('cuda', torch.bfloat16)
    
    def sample_batch_with_logprob(self, batch_vid, batch_mouse, batch_btn):
        """
        Sample a batch of videos using the current model with log probability tracking.
        
        :param batch_vid: Input video frames 
        :param batch_mouse: Mouse movement data
        :param batch_btn: Button press data
        :return: Dictionary containing samples with latents, log_probs, etc.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get EMA model for sampling
            def get_ema_core():
                if self.world_size > 1:
                    return self.ema.ema_model.module.core
                else:
                    return self.ema.ema_model.core
            
            model = get_ema_core()
            batch_size = batch_vid.shape[0]
            
            # Initialize with noise (rectified flow starts from noise)
            x = torch.randn_like(batch_vid)
            ts = torch.ones(batch_size, device=x.device, dtype=x.dtype)
            dt = 1.0 / self.sampling_steps
            
            # Store trajectory data for DDPO
            all_latents = [x.clone()]
            all_log_probs = []
            all_timesteps = [ts.clone()]
            all_preds = []
            
            # Sampling loop with log probability tracking
            for step in range(self.sampling_steps):
                with self.ctx:
                    # Get model prediction
                    pred = model(x, ts, batch_mouse, batch_btn)
                    all_preds.append(pred.clone())
                    
                    # Rectified flow update: x = x - pred * dt
                    if step < self.sampling_steps - 1:
                        next_x = x - pred * dt
                        next_ts = ts - dt
                        
                        # Compute log probability of the transition
                        # For rectified flow, this is the log probability of next_x given current state
                        # Simplified computation - can be made more sophisticated
                        diff = next_x - (x - pred * dt)
                        log_prob = -0.5 * torch.sum(diff ** 2, dim=(1, 2, 3, 4))
                        
                        all_latents.append(next_x.clone())
                        all_log_probs.append(log_prob)
                        all_timesteps.append(next_ts.clone())
                        
                        x = next_x
                        ts = next_ts
            
            # Stack trajectory data
            all_latents = torch.stack(all_latents, dim=1)  # (batch, timesteps+1, ...)
            all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch, timesteps)
            all_timesteps = torch.stack(all_timesteps[:-1], dim=1)  # (batch, timesteps)
            all_preds = torch.stack(all_preds, dim=1)  # (batch, timesteps, ...)
            
            return {
                'latents': all_latents[:, :-1],  # Remove last timestep
                'next_latents': all_latents[:, 1:],  # Remove first timestep  
                'log_probs': all_log_probs,
                'timesteps': all_timesteps,
                'predictions': all_preds,
                'final_latents': x,
                'mouse': batch_mouse,
                'buttons': batch_btn,
            }
    
    def compute_rewards(self, final_latents, mouse_data, button_data):
        """
        Compute rewards for generated videos.
        
        :param final_latents: Final latent representations
        :param mouse_data: Mouse movement data
        :param button_data: Button press data
        :return: Tensor of rewards
        """
        if self.reward_fn is None:
            raise ValueError("Reward function not set. Use set_reward_function() to set it.")
        
        # Decode latents to videos
        with torch.no_grad():
            videos = self.decode_fn(final_latents * self.train_cfg.vae_scale)
        
        # Compute rewards asynchronously
        rewards_future = self.executor.submit(self.reward_fn, videos, mouse_data, button_data)
        rewards = rewards_future.result()
        
        return torch.tensor(rewards, device=final_latents.device)
    
    def compute_advantages(self, rewards):
        """Compute advantages for PPO."""
        # Simple advantage computation - can be enhanced with value function
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages
    
    def ppo_loss(self, log_probs, old_log_probs, advantages):
        """
        Compute PPO clipped loss.
        
        :param log_probs: Current policy log probabilities
        :param old_log_probs: Old policy log probabilities  
        :param advantages: Computed advantages
        :return: PPO loss
        """
        # Importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped advantages
        advantages = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        
        # PPO loss
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        
        return loss
    
    def train_step(self, samples, local_step):
        """
        Perform one training step using PPO.
        
        :param samples: Dictionary containing trajectory samples
        :param local_step: Current local step for gradient accumulation
        :return: Dictionary of training metrics
        """
        self.model.train()
        
        batch_size, num_timesteps = samples['timesteps'].shape
        info = defaultdict(list)
        
        # Train on subset of timesteps (randomly sample for efficiency)
        train_timesteps = min(num_timesteps, self.num_train_timesteps)
        timestep_indices = torch.randperm(num_timesteps)[:train_timesteps]
        
        for t_idx in timestep_indices:
            # Get current timestep data
            latents = samples['latents'][:, t_idx]
            next_latents = samples['next_latents'][:, t_idx]
            timesteps = samples['timesteps'][:, t_idx]
            old_log_probs = samples['log_probs'][:, t_idx]
            old_predictions = samples['predictions'][:, t_idx]
            advantages = samples['advantages']
            
            with self.ctx:
                # Forward pass - get current model prediction
                model = self.get_module()
                current_pred = model(latents, timesteps, samples['mouse'], samples['buttons'])
                
                # Compute log probability under current policy
                # For rectified flow, the log prob is related to how well we predict the flow
                dt = 1.0 / self.sampling_steps
                expected_next = latents - current_pred * dt
                diff = next_latents - expected_next
                log_prob = -0.5 * torch.sum(diff ** 2, dim=(1, 2, 3, 4))
                
                # Compute PPO loss
                loss = self.ppo_loss(log_prob, old_log_probs, advantages) / self.accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Compute metrics
            with torch.no_grad():
                ratio = torch.exp(log_prob - old_log_probs)
                info['approx_kl'].append(0.5 * torch.mean((log_prob - old_log_probs) ** 2))
                info['clipfrac'].append(torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()))
                info['ppo_loss'].append(loss * self.accum_steps)
                info['log_prob'].append(log_prob.mean())
                info['old_log_prob'].append(old_log_probs.mean())
        
        # Optimization step (if gradient accumulation is complete)
        if local_step % self.accum_steps == 0:
            # Gradient clipping
            if self.train_cfg.opt.lower() != "muon":
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.opt)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            self.ema.update()
        
        # Average metrics
        if info:
            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
        
        return info
    
    def train_epoch(self, epoch, loader):
        """Train for one epoch using DDPO."""
        if self.reward_fn is None:
            raise ValueError("Reward function not set. Use set_reward_function() to set it.")
        
        # Sampling phase
        self.model.eval()
        all_samples = []
        all_rewards = []
        
        for batch_idx, (batch_vid, batch_mouse, batch_btn) in enumerate(tqdm(
            loader,
            desc=f"Epoch {epoch}: Sampling",
            disable=self.rank != 0
        )):
            # Prepare batch data
            batch_vid = batch_vid.cuda().bfloat16() / self.train_cfg.vae_scale
            batch_mouse = batch_mouse.cuda().bfloat16()
            batch_btn = batch_btn.cuda().bfloat16()
            
            # Sample trajectories with log probabilities
            samples = self.sample_batch_with_logprob(batch_vid, batch_mouse, batch_btn)
            
            # Compute rewards
            rewards = self.compute_rewards(samples['final_latents'], batch_mouse, batch_btn)
            samples['rewards'] = rewards
            
            all_samples.append(samples)
            all_rewards.extend(rewards.cpu().numpy())
            
            # Limit number of batches for sampling
            if batch_idx >= self.num_batches_per_epoch - 1:
                break
        
        # Compute advantages
        all_rewards = np.array(all_rewards)
        advantages = self.compute_advantages(torch.tensor(all_rewards))
        
        # Add advantages to samples
        start_idx = 0
        for samples in all_samples:
            end_idx = start_idx + len(samples['rewards'])
            samples['advantages'] = advantages[start_idx:end_idx].to(f'cuda:{self.local_rank}')
            start_idx = end_idx
        
        # Training phase - multiple inner epochs over collected data
        epoch_info = defaultdict(list)
        local_step = 0
        
        for inner_epoch in range(self.num_inner_epochs):
            # Shuffle samples for training
            shuffled_samples = all_samples.copy()
            np.random.shuffle(shuffled_samples)
            
            for samples in tqdm(
                shuffled_samples,
                desc=f"Epoch {epoch}.{inner_epoch}: Training",
                disable=self.rank != 0
            ):
                info = self.train_step(samples, local_step)
                local_step += 1
                
                # Collect metrics
                for k, v in info.items():
                    epoch_info[k].append(v)
        
        # Log metrics
        if self.rank == 0:
            log_dict = {
                'epoch': epoch,
                'reward_mean': np.mean(all_rewards),
                'reward_std': np.std(all_rewards),
                'num_samples': len(all_rewards),
            }
            
            # Add training metrics
            for k, v in epoch_info.items():
                if len(v) > 0:
                    if isinstance(v[0], torch.Tensor):
                        log_dict[k] = torch.mean(torch.stack(v)).item()
                    else:
                        log_dict[k] = np.mean(v)
            
            if self.logging_cfg is not None:
                wandb.log(log_dict, step=self.total_step_counter)
            
            print(f"Epoch {epoch}: Reward {log_dict['reward_mean']:.3f} � {log_dict['reward_std']:.3f}")
        
        self.total_step_counter += 1
        
        return epoch_info
    
    def save(self):
        save_dict = {
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)
    
    def load(self):
        has_ckpt = False
        try:
            if self.train_cfg.resume_ckpt is not None:
                save_dict = super().load(self.train_cfg.resume_ckpt)
                has_ckpt = True
        except:
            print("Error loading checkpoint")
        
        if not has_ckpt:
            return
        
        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        """Main training loop."""
        if self.rank == 0:
            print("Starting DDPO training...")
        
        # Setup training components
        self.setup_training()
        
        # Load checkpoint if available
        self.load()
        
        # Setup data loader
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        
        # Timer and metrics
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')
        
        # Training loop
        for epoch in range(self.train_cfg.epochs):
            self.barrier()
            
            # Train for one epoch
            self.train_epoch(epoch, loader)
            
            # Save checkpoint
            if epoch % self.train_cfg.get('save_interval', 10) == 0 and self.rank == 0:
                self.save()
                print(f"Saved checkpoint at epoch {epoch}")
        
        # Cleanup
        self.executor.shutdown(wait=True)
        
        if self.rank == 0:
            print("DDPO training completed!")
