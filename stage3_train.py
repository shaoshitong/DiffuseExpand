"""
Train a noised image classifier on Segmentation Dataset.
"""

import argparse
import os
import blobfile as bf
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np

from utils import set_device, setup_dist, create_model_and_diffusion, create_named_schedule_sampler, TrainLoop, \
    create_classifier_and_diffusion, PSNRLoss, DiceLoss
from backbone.fp16_util import MixedPrecisionTrainer

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='ISIC', help='dataset')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
parser.add_argument('--learn_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training networks')
parser.add_argument('--data_path', type=str,
                    default='./covid-chestxray-dataset/images/',
                    help='dataset path')
parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
parser.add_argument('--csv_path', type=str,
                    default="./covid-chestxray-dataset/metadata.csv")
parser.add_argument('--save_path', type=str, default="./stage2")
parser.add_argument('--unet_ckpt_path', type=str,
                    default="/home/sst/product/diffusion-model-learning/demo/256x256_diffusion.pt")
parser.add_argument('--class_cond', type=bool, default=True)
parser.add_argument('--num_classes_1', type=int, default=2)
parser.add_argument('--num_classes_2', type=int, default=-1)
parser.add_argument('--cuda_devices', type=str, default="0", help="data parallel training")


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def yield_data(dataloader):
    while True:
        yield from dataloader


def create_argparser():
    defaults = dict(
        iterations=5000,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        data_dir="",
        val_data_dir="",
        noised=True,
        weight_decay=0.0,
        anneal_lr=False,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint=None,
        log_interval=10,
        eval_interval=5,
        save_interval=1000,
        channel_mult="",
        lr=3e-4,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=30000,
    )

    diffusion_defaults = dict(
        learn_sigma=False,  # TODO; MUST BE FALSE
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    defaults.update(diffusion_defaults)

    # TODO: classifier is not need
    classifier_defaults = dict(
        image_size=256,
        classifier_use_fp16=True,
        classifier_width=64,
        classifier_depth=2,
        classifier_attention_resolutions="16",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )
    defaults.update(classifier_defaults)

    add_dict_to_argparser(parser, defaults)
    return parser


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def set_random_seed(number=0):
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import random
    import numpy as np
    np.random.seed(number)
    random.seed(number)


# def load_model(model_dict, model):
#     model_state_dict = model.state_dict()
#     pretrained_dict = {
#         k: v
#         for k, v in model_dict.items()
#         if k in model_state_dict and v.shape == model_state_dict[k].shape
#     }
#     print(
#         f"the prune number is {round((len(model_state_dict.keys())-len(pretrained_dict.keys()))*100/len(model_state_dict.keys()),3)}%"
#     )
#     print("missing keys:")
#     for key in model_state_dict.keys():
#         if key not in pretrained_dict:
#             print(key)
#     model_state_dict.update(pretrained_dict)
#     model.load_state_dict(model_state_dict)
#     return model


def main_worker(gpu, args, ngpus_per_node, world_size, dist_url):
    # TODO: Initialize the ddp environment
    print("Use GPU: {} for training".format(gpu))
    rank = 0
    dist_backend = "nccl"
    rank = rank * ngpus_per_node + gpu
    print("world_size:", world_size)
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )

    set_random_seed(rank + np.random.randint(0, 1000))
    torch.cuda.set_device(gpu)

    # TODO: build dataset
    print("build dataset....")
    if args.dataset == "COVID19":
        from utils.covid19_dataset import COVID19Dataset, clean_dataset
        assert args.csv_path != "no", "COVID-19 Segmentation task need csv metadata!"
        dst = COVID19Dataset(imgpath=args.data_path, csvpath=args.csv_path, semantic_masks=True)
        dst = clean_dataset(dst)
    elif args.dataset == "ISIC":
        from utils.isic_dataset import SkinDataset
        image_root = '{}/data_train.npy'.format(args.data_path)
        gt_root = '{}/mask_train.npy'.format(args.data_path)
        dst = SkinDataset(image_root=image_root, gt_root=gt_root)
    else:
        raise NotImplementedError
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = [0 for i in range(len(dst))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    dst_train = torch.utils.data.Subset(dst, train_indices)
    dst_test = torch.utils.data.Subset(dst, valid_indices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dst_train)
    train_loader = DataLoader(
        dst_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=(torch.cuda.is_available()),
    )
    test_loader = DataLoader(
        dst_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(torch.cuda.is_available()),
    )
    NAME = [
        "image_size",
        "classifier_use_fp16",
        "classifier_width",
        "classifier_depth",
        "classifier_attention_resolutions",
        "classifier_use_scale_shift_norm",
        "classifier_resblock_updown",
        "classifier_pool",
        "learn_sigma",
        "diffusion_steps",
        "noise_schedule",
        "timestep_respacing",
        "use_kl",
        "predict_xstart",
        "rescale_timesteps",
        "rescale_learned_sigmas",
        "num_classes_1",
        "num_classes_2",
        "isic"
    ]

    # TODO: Define UNet and diffusion scheduler
    args.num_classes_2 = 1
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, NAME)
    )

    # TODO: build a sampler (default is uniform)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # TODO: training
    print("begin training....")
    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )
    model = DDP(
        model.cuda(gpu),
        device_ids=[gpu],
        output_device=gpu,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)

    psnr_loss = PSNRLoss()
    dice_loss = DiceLoss()

    def split_microbatches(microbatch, *args):
        bs = len(args[0])
        if microbatch == -1 or microbatch >= bs:
            yield tuple(args)
        else:
            for i in range(0, bs, microbatch):
                yield tuple(x[i: i + microbatch] if x is not None else None for x in args)

    def forward_backward_log(data_loader, prefix="train"):
        batch, cond2 = data_loader
        labels = cond2.cuda(gpu).float()
        batch = batch.cuda(gpu)
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], gpu)
            batch = diffusion.q_sample(batch, t)
        else:
            t = torch.zeros(batch.shape[0], dtype=torch.long).cuda(gpu)

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            diceloss = dice_loss(logits.sigmoid(), sub_labels)
            mseloss = F.l1_loss(logits.sigmoid(), sub_labels)
            loss = diceloss + mseloss
            losses = {}
            losses[f"{prefix}_dice_loss"] = diceloss.detach().item()
            losses[f"{prefix}_l1_loss"] = mseloss.detach().item()
            losses[f"{prefix}_psnr_loss"] = psnr_loss(logits.sigmoid(), sub_labels).detach().item()
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad(opt)
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
            return losses
    for step in range(int(args.iterations//len(train_loader))):
        for i,(batch,cond2) in enumerate(train_loader): 
            if gpu==0:
                print(f"step is {step*len(train_loader)+i}")
            if args.anneal_lr:
                set_annealed_lr(opt, args.lr, (step) / args.iterations)

            forward_backward_log([batch,cond2])
            mp_trainer.optimize(opt)
            if (
                    step
                    and dist.get_rank() == 0
                    and not (step) % args.save_interval
            ):
                print("saving model...")
                save_model(mp_trainer, opt, step,"./stage2/")
        total_loss = {"val_dice_loss":0,"val_psnr_loss":0,"val_l1_loss":0}
        for i,(batch,cond2) in enumerate(test_loader):
            with torch.no_grad():
                with model.no_sync():
                    model.eval()
                    losses = forward_backward_log([batch,cond2], prefix="val")
                    for key in total_loss.keys():
                        total_loss[key] += losses[key]
                    model.train()
        for key in total_loss.keys():
            total_loss[key] /= len(test_loader)
        if gpu==0:
            print(total_loss)

    if dist.get_rank() == 0:
        save_model(mp_trainer, opt, args.iterations,args.save_path)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, save_path):
    if dist.get_rank() == 0:
        global args
        torch.save(
            mp_trainer.master_params,
            os.path.join(save_path, f"stage3_isic_model_{step}.pt"),
        )


def main():
    args = create_argparser().parse_args()
    if args.dataset == "ISIC":
        args.isic = True
    parallel_function = setup_dist(args)
    parallel_function(main_worker)


if __name__ == "__main__":
    main()
