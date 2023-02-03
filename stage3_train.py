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

from utils import set_device, setup_dist, create_model_and_diffusion, create_named_schedule_sampler,TrainLoop, create_classifier_and_diffusion
from backbone.fp16_util import MixedPrecisionTrainer

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='COVID19', help='dataset')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training networks')
parser.add_argument('--data_path', type=str, default='/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/', help='dataset path')
parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
parser.add_argument('--csv_path', type=str, default="/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv")
parser.add_argument('--save_path', type=str, default="/home/Bigdata/mtt_distillation_ckpt/stage2")
parser.add_argument('--unet_ckpt_path', type=str, default="/home/sst/product/diffusion-model-learning/demo/256x256_diffusion.pt")
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


def create_argparser():
    defaults = dict(
        iterations=150000,
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
        save_interval=100,
        channel_mult="",
        lr=3e-4,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=30000,
    )

    diffusion_defaults = dict(
        learn_sigma=False, # TODO; MUST BE FALSE
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
    classifier_defaults=dict(
            image_size=256,
            classifier_use_fp16=True,
            classifier_width=64,
            classifier_depth=2,
            classifier_attention_resolutions="",  # 16
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
    from utils.covid19_dataset import COVID19Dataset, generate_clean_dataset, clean_dataset
    assert args.csv_path != "no", "COVID-19 Segmentation task need csv metadata!"
    dst = COVID19Dataset(imgpath=args.data_path, csvpath=args.csv_path, semantic_masks=True)
    train_set = generate_clean_dataset(dst)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=(torch.cuda.is_available()),
    )
    iter = 0
    for i,(image,label1,label2) in enumerate(train_set):
        if label1 == 0:
            torchvision.transforms.ToPILImage()(image).save(f"{i}_image.png")
        else:
            torchvision.transforms.ToPILImage()(image).save(f"{i}_mask.png")
    exit(-1)
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
    ]
    # TODO: Define UNet and diffusion scheduler
    args.num_classes_2 = int(len(train_set)//2)
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
        model,
        device_ids=[gpu],
        output_device=gpu,
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)



def main():
    args = create_argparser().parse_args()
    parallel_function = setup_dist(args)
    parallel_function(main_worker)


if __name__ == "__main__":
    main()
