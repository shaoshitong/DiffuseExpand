import copy
import functools
import os
import warnings

import torch
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from backbone.nn import update_ema
from .schedule_sampler import LossAwareSampler, UniformSampler
from backbone.fp16_util import MixedPrecisionTrainer
from .dist_utils import sync_params

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def yield_data(dataloader):
    while True:
        yield from dataloader


class TrainLoop:
    def __init__(
            self,
            *,
            gpu,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            save_interval,
            save_path,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
    ):
        self.gpu = gpu
        self.model = model
        self.diffusion = diffusion
        self.train_data = data
        self.data = yield_data(self.train_data)
        self.batch_size = batch_size
        self.save_path = save_path
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        # self._load_optimizer_state()
        # self._resume_parameters()

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model.cuda(gpu),
                device_ids=[gpu],
                output_device=gpu,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )
        else:
            if dist.get_world_size() > 1:
                warnings.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        print(resume_checkpoint)
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint, map_location="cuda"
                )
            )
        # sync_params(self.model.parameters())

    def _resume_parameters(self):
        resume_checkpoint = os.path.join(self.save_path, f"model_stage2_10000.pt")
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint, map_location="cuda"
                )
            )
        # sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        opt_checkpoint = os.path.join(
            self.save_path, f"opt_stage2_10000.pt"
        )
        if os.path.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location="cuda"
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond1, cond2 = next(self.data)
            self.run_step(batch, cond1, cond2)
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond1, cond2):
        self.forward_backward(batch, cond1, cond2)
        took_step = self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond1, cond2):
        self.mp_trainer.zero_grad(self.opt)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].cuda(self.gpu)*2-1
            micro_cond = {"y1": cond1[i: i + self.microbatch].cuda(self.gpu),
                          "y2": cond2[i: i + self.microbatch].cuda(self.gpu)}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.gpu)
            with torch.cuda.amp.autocast():
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            print({k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        if self.gpu==0 and self.step%100==0:
            print(f"now lr is {lr}")
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        print("step", self.step + self.resume_step)
        print("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            if self.gpu == 0:
                state_dict = params
                print(f"saving model {rate}...")
                filename = f"model_stage2_{self.resume_step+self.step}.pt"
                th.save(state_dict, os.path.join(self.save_path, filename))

        save_checkpoint(0, self.mp_trainer.model.state_dict())
        # if self.gpu == 0:
        #     filename = f"opt_stage2_{self.resume_step+self.step}.pt"
        #     th.save(self.opt.state_dict(), os.path.join(self.save_path, filename))
        print("finish saving!")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None
