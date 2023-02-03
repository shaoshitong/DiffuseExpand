from utils import NoiseScheduleVP, model_wrapper, DPM_Solver
import argparse
from torch.utils.data import DataLoader
from utils import create_model_and_diffusion
import torch

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='COVID19', help='dataset')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training networks')
parser.add_argument('--data_path', type=str,
                    default='./covid-chestxray-dataset/images/',
                    help='dataset path')
parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
parser.add_argument('--csv_path', type=str,
                    default="./covid-chestxray-dataset/metadata.csv")
parser.add_argument('--save_path', type=str, default="/home/Bigdata/mtt_distillation_ckpt/stage2")
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


def create_argparser():
    defaults = dict(
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
        lr=1e-4,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=300,
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
    # classifier_defaults=dict(
    #         image_size=64,
    #         classifier_use_fp16=False,
    #         classifier_width=128,
    #         classifier_depth=2,
    #         classifier_attention_resolutions="32,16,8",  # 16
    #         classifier_use_scale_shift_norm=True,  # False
    #         classifier_resblock_updown=True,  # False
    #         classifier_pool="attention",
    #     )
    # defaults.update(classifier_defaults)

    add_dict_to_argparser(parser, defaults)
    return parser


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


# TODO: Definition

args = create_argparser().parse_args()
BATCHSIZE = 6
label1 = torch.ones(BATCHSIZE).long().cuda()
label1[::2]  = label1[::2] * 0
label2 = torch.ones(BATCHSIZE).long().cuda() * 5

NAME = [
    "image_size",
    "class_cond",
    "learn_sigma",
    "num_channels",
    "num_res_blocks",
    "channel_mult",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "attention_resolutions",
    "dropout",
    "diffusion_steps",
    "noise_schedule",
    "timestep_respacing",
    "use_kl",
    "predict_xstart",
    "rescale_timesteps",
    "rescale_learned_sigmas",
    "use_checkpoint",
    "use_scale_shift_norm",
    "resblock_updown",
    "use_fp16",
    "use_new_attention_order",
    "num_classes_1",
    "num_classes_2",
]

# TODO: Define UNet and diffusion scheduler
from utils.covid19_dataset import COVID19Dataset, generate_clean_dataset

assert args.csv_path != "no", "COVID-19 Segmentation task need csv metadata!"
dst = COVID19Dataset(imgpath=args.data_path, csvpath=args.csv_path, semantic_masks=True)
train_set = generate_clean_dataset(dst)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=(torch.cuda.is_available()),
)

# TODO: I. define model
args.num_classes_2 = int(len(train_set) // 2)
_model_fn, diffusion = create_model_and_diffusion(
    **args_to_dict(args, NAME)
)
import os, sys

model_path = "./stage2/model_stage2_9000.pt"
if not os.path.exists(model_path):
    raise KeyError

_model_fn.load_state_dict(torch.load(model_path, map_location="cpu"))
_model_fn.cuda()


@torch.no_grad()
def model(x, t, **kwargs):
    B, C = x.shape[:2]
    model_output = _model_fn(x, t, **kwargs)
    return model_output


# TODO: II. define model_kwargs
model_kwargs = {"y1": label1, "y2": label2}

# TODO: III. define condition
tau = 1.0
condition = None

# TODO: IV. define unconditional_condition
unconditional_condition = None  # Nothing to do with guidance-classifier scenarios

# TODO: V. define guidance_scale
guidance_scale = 1.  # Nothing to do with uncond scenarios
# TODO: VI. define classifier
classifier = None  # Nothing to do with uncond scenarios
# TODO: VII. define classifier_kwargs
classifier_kwargs = {}  # Nothing to do with uncond scenarios

# TODO: VIII. define betas
from utils import get_named_beta_schedule

betas = torch.from_numpy(get_named_beta_schedule("linear", 1000)).cuda()

noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type="noise",  # or "x_start" or "v" or "score"
    model_kwargs=model_kwargs,
    guidance_type="uncond",
    condition=condition,
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    classifier_kwargs=classifier_kwargs,
)
dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",correcting_x0_fn = "dynamic_thresholding")
image_shape = (BATCHSIZE//2, 1, 256, 256)

for j in range(args.num_classes_2):
    label2 = torch.ones(BATCHSIZE).long().cuda() * j
    model_kwargs =  {"y1": label1, "y2": label2}
    model_fn = model_wrapper(
                model,
                    noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                            model_kwargs=model_kwargs,
                                guidance_type="uncond",
                                    condition=condition,
                                        guidance_scale=guidance_scale,
                                            classifier_fn=classifier,
                                                classifier_kwargs=classifier_kwargs,
                                                )
    x_T = torch.randn(image_shape).unsqueeze(1).expand(-1,2,1,256,256).cuda().view(BATCHSIZE,1,256,256)
    with torch.no_grad():
        x_sample = dpm_solver.sample(
        x_T,
        steps=30,
        order=3,
        skip_type="time_uniform",
        method="multistep",
        )

    from PIL import Image


    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    for i in range(x_sample.shape[0]):
        sub_image = x_sample[i]
        print(sub_image.min(),sub_image.max())
        sub_image = (sub_image / 2 + 0.5).clamp(0, 1)
        sub_image = sub_image.cpu().permute(1, 2, 0).numpy()
        numpy_to_pil(sub_image)[0].save(f"sample_{int(BATCHSIZE//2)*j+int(i//2)}_{'image' if label1[i].item()==0 else 'mask'}_{label2[i].item()}.png")
