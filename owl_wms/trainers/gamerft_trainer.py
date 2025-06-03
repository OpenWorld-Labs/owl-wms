"""
Trainer for reconstruction only
"""

import torch
from ema_pytorch import EMA
import wandb
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import einops as eo

from .base import BaseTrainer

from ..utils import freeze, Timer, find_unused_params
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class RFTTrainer(BaseTrainer):
    """
    Trainer for rectified flow transformer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):  
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0
        self.decoder = get_decoder_only()
        freeze(self.decoder)

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
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
        torch.cuda.set_device(self.local_rank)

        # Prepare model and ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.decoder = self.decoder.cuda().eval().bfloat16()
        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(
            self.model,
            beta = 0.999,
            update_after_step = 0,
            update_every = 1
        )

        def get_ema_core():
            if self.world_size > 1:
                return self.ema.ema_model.module.core
            else:
                return self.ema.ema_model.core

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda',torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')
        
        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch_vid, batch_mouse, batch_btn in loader:
                batch_vid = batch_vid.cuda().bfloat16() / self.train_cfg.vae_scale
                batch_mouse = batch_mouse.cuda().bfloat16()
                batch_btn = batch_btn.cuda().bfloat16()

                with ctx:
                    loss = self.model(batch_vid,batch_mouse,batch_btn) / accum_steps

                self.scaler.scale(loss).backward()
                #find_unused_params(self.model)

                metrics.log('diffusion_loss', loss)

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    if self.train_cfg.opt.lower() != "muon":
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        # Sampling commented out for now
                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx, torch.no_grad():
                                n_samples = self.train_cfg.n_samples
                                samples, sample_mouse, sample_button = sampler(
                                    get_ema_core(),
                                    batch_vid[:n_samples],
                                    batch_mouse[:n_samples],
                                    batch_btn[:n_samples],
                                    sampling_steps=30,
                                    decode_fn = decode_fn,
                                    scale=self.train_cfg.vae_scale
                                ) # -> [b,n,c,h,w]
                                wandb_dict['samples'] = to_wandb(samples, sample_mouse, sample_button)
                            

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()
                        
                    self.barrier()